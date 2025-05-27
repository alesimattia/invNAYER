import os, glob, random, shutil, warnings, subprocess, registry, datafree, argparse, pickle, time
from math import gamma

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import wandb

from datafree.metrics.generated_img_quality import inception_score_from_folder
from datafree.metrics.PCA import model_PCA
from datafree.metrics.TSNE import compute_TSNE
from datafree.metrics.model_comparator import Comparator
from datafree.metrics.confusionMatrix import compute_confusion_matrix


parser = argparse.ArgumentParser(description='Inversion loss NAYER')

# Da "imagenet_inversion.py" coefficient per loss_aux
parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
parser.add_argument('--l2', type=float, default=0.00001, help='l2 loss on the image')
parser.add_argument('--r_feature', type=float, default=0.05, help='coefficient for feature distribution regularization')
parser.add_argument('--first_bn_multiplier', type=float, default=10., help='additional multiplier on first bn layer of R_feature')
parser.add_argument('--main_loss_multiplier', type=float, default=1.0, help='coefficient for the main loss in optimization')
parser.add_argument('--adi_scale', type=float, default=0.0, help='Coefficient for Adaptive Deep Inversion')
# Metriche
parser.add_argument('--metrics', nargs='+', help='Lista delle metriche da calcolare: PCA, TSNE, distance, DICE, confusionMatrix, JSindex', 
                    default=["PCA", "TSNE", "distance", "DICE", "confusionMatrix", "JSindex"],
                    choices=["PCA", "TSNE", "distance", "DICE", "confusionMatrix", "JSindex"])
# Modelli preaddestrati
parser.add_argument('--nayer_student', type=str, default="best_c10r34r18-tvL2-0.0005__l2-0.00001", 
                    help='Path modello .pth preaddestrato con NAYER classico; per fare poi train di KD_student')
parser.add_argument('--scratch_student', type=str, default='cifar10_resnet18_100ep', 
                    help='Path modello .pth addestrato con train_scratch.py')
parser.add_argument('--KD_student', type=str, default='KD_student_best_c10r34r18-tvL2-0.0005__l2-0.00001',
                    help='Path modello .pth che combini predizioni insegnante e migliore modello NAYER')
parser.add_argument('--train_distilled_student', default=False, action=argparse.BooleanOptionalAction, help='Addestra uno studente distillato con BCE+KL; richiede --nayer_student')
parser.add_argument('--alpha', default=0.2, type=float, help='Bilanciamento tra BCE loss (teacher) e KL loss (nayer student) per KD_student; richiede --train_distilled_student')


# Data Free
parser.add_argument('--method', default='nldf')
parser.add_argument('--adv', default=1.33, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--bn', default=10, type=float, help='scaling factor for BN regularization')
parser.add_argument('--oh', default=0.5, type=float, help='scaling factor for one hot loss (cross entropy)')

# TODO: optimize contr. now set to 0.5
parser.add_argument('--contr', default=0.5, type=float, help='scaling factor for contrastive loss (augmentation-based)')
parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
parser.add_argument('--balance', default=0, type=float, help='scaling factor for class balance')
parser.add_argument('--save_dir', default='run/', type=str)
parser.add_argument('--cmi_init', default=None, type=str, help='path to pre-inverted data')

parser.add_argument('--bn_mmt', default=0.9, type=float,
                    help='momentum when fitting batchnorm statistics')

# CDF
parser.add_argument('--log_tag', default='ti-r34-test')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup', default=1, type=int, metavar='N',
                    help='which epoch to start kd')

parser.add_argument('--g_steps', default=5, type=int, metavar='N',
                    help='number of iterations for generation')
parser.add_argument('--g_life', default=10, type=int,
                    help='meta gradient: is maml or reptile')
parser.add_argument('--g_loops', default=1, type=int, metavar='N', help='which epoch to start kd')
parser.add_argument('--gwp_loops', default=1, type=int, metavar='N', help='which epoch to start kd')
parser.add_argument('--lr_g', default=4e-3, type=float, help='initial learning rate for generator')
parser.add_argument('--le_emb_size', default=1000, type=int, metavar='N', help='which epoch to start kd')
parser.add_argument('--bnt', default=20, type=float,
                    help='momentum when fitting batchnorm statistics')
parser.add_argument('--oht', default=3.0, type=float,
                    help='momentum when fitting batchnorm statistics')

# Basic
# FIXME: path without ".."
parser.add_argument('--data_root', default='../')
parser.add_argument('--teacher', default='resnet34')
parser.add_argument('--student', default='resnet18')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tiny_imagenet', 'imagenet'])
parser.add_argument('--lr', default=0.2, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--eta_min', default=2e-4, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--T', default=20, type=float)

parser.add_argument('--kd_steps', default=400, type=int, metavar='N',
                    help='number of iterations for KD after generation')
parser.add_argument('--ep_steps', default=400, type=int, metavar='N',
                    help='number of total iterations in each epoch')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Device
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
# TODO: Distributed and FP-16 training 
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--fp16', action='store_true',
                    help='use fp16')

# Misc
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--wandb', default='offline', type=str)

best_acc1 = 0
time_cost = 0


def main():
    args = parser.parse_args()

    args.save_dir = args.save_dir + args.log_tag
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        teacher, student, synthesizer, evaluator = main_worker(args.gpu, ngpus_per_node, args)
        
        '''
        Eseguito al termine delle epoche di addestramento => cartella popolata con immagini sintetizzate
        '''
        if len( glob.glob(os.path.join("./", args.save_dir, "*.png")) ) > 0:
            inception_mean, inception_std = inception_score_from_folder(args.save_dir)
            args.logger.info(f"Inception Score: {inception_mean:.4f} ± {inception_std:.4f}")
            wandb.log({"Inception Score": inception_mean, "Inception Std": inception_std})
        #Altrimenti salta

    #
    #
    #
    ### VALUTAZIONE MODELLI
    #
    if any(m in args.metrics for m in ["PCA", "TSNE", "distance", "DICE", "confusionMatrix", "JSindex"]):
        num_classes, _, _ = registry.get_dataset(name=args.dataset, data_root=args.data_root)
        dataset_location = os.path.join(os.path.dirname(__file__), '../', args.dataset.upper())

        #Caricamento modelli pre-addestrati
        teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
        teacher.load_state_dict(torch.load(f'./checkpoints/pretrained/{args.dataset}_{args.teacher}.pth', map_location='cpu')['state_dict'])
                    # Stessa architettura di args.student => resnet18
        nayerStudent = registry.get_model(args.student, num_classes=num_classes, pretrained=True).eval() #best_c10r34r18-tvL2-0.0005__l2-0.00001
        nayerStudent.load_state_dict(torch.load(f'./checkpoints/datafree-{args.method}/cifar10-resnet34-resnet18--{args.nayer_student}.pth', map_location='cpu')['state_dict']) 
                    # Stessa architettura di args.student => resnet18
        scratchStudent = registry.get_model(args.student, num_classes=num_classes, pretrained=True).eval() #epoche di esecuzione != epoche train studente scratch 
        scratchStudent.load_state_dict(torch.load(f'./checkpoints/scratch/{args.dataset}_{args.student}_100ep.pth', map_location='cpu')['state_dict'])
                    # Stessa architettura di args.student => resnet18
        KDstudent = registry.get_model(args.student, num_classes=num_classes, pretrained=True).eval()
        KDstudent.load_state_dict(torch.load(f'./checkpoints/datafree-{args.method}/cifar10-resnet34-resnet18--{args.KD_student}.pth', map_location='cpu')['state_dict'])


        if "PCA" in args.metrics:
            components = 2
            teacher_img = model_PCA(teacher, components=components, batch_size=args.batch_size, num_workers=args.workers, 
                    dataset_root=dataset_location,
                    output_path=f"./PCA_img/teacher_PCA.png")

            nayerStudent_img = model_PCA(nayerStudent, components=components, batch_size=args.batch_size, num_workers=args.workers,
                    dataset_root=dataset_location,
                    output_path=f"./PCA_img/{args.nayer_student}_PCA.png")

            scratchStudent_img = model_PCA(scratchStudent, components=components, batch_size=args.batch_size, num_workers=args.workers,
                    dataset_root=dataset_location,
                    output_path=f"./PCA_img/scratchStudent_PCA.png")

            kdStudent_img = model_PCA(KDstudent, components=components, batch_size=args.batch_size, num_workers=args.workers,
                    dataset_root=dataset_location,
                    output_path=f"./PCA_img/{args.KD_student}_PCA.png")
            
            wandb.log({
                "PCA - Teacher": wandb.Image(teacher_img),
                "PCA - NAYER Student": wandb.Image(nayerStudent_img),
                "PCA - Scratch Student": wandb.Image(scratchStudent_img),
                "PCA - KD Student": wandb.Image(kdStudent_img)
            })


        teacher_nayerStud_Comparator = Comparator(teacher, nayerStudent, dataset_location, args.batch_size, args.workers)
        kdStud_nayerStud_Comparator = Comparator(KDstudent, nayerStudent, dataset_location, args.batch_size, args.workers)
        scratchStud_nayerStud_Comparator = Comparator(scratchStudent, nayerStudent, dataset_location, args.batch_size, args.workers)

        if "distance" in args.metrics: 
            teacher_student_dst = teacher_nayerStud_Comparator.prediction_distance()
            scratchStudent_nayerStudent_dst = teacher_nayerStud_Comparator.prediction_distance()
            kdStudent_nayerStudent_dst = teacher_nayerStud_Comparator.prediction_distance()
            kdStudent_scratchStudent_dst = teacher_nayerStud_Comparator.prediction_distance()
            
            wandb.log({
                "Prediction Distance Teacher‑NAYER Student (per class)": wandb.Histogram([teacher_student_dst[k] for k in sorted(teacher_student_dst.keys())]),
                "Prediction Distance NAYER Student-Scratch Student (per class)": wandb.Histogram([scratchStudent_nayerStudent_dst[k] for k in sorted(scratchStudent_nayerStudent_dst.keys())]),
                "Prediction Distance KD student-NAYER Student (per class)": wandb.Histogram([kdStudent_nayerStudent_dst[k] for k in sorted(kdStudent_nayerStudent_dst.keys())]),
                "Prediction Distance KD student-Scratch Student (per class)": wandb.Histogram([kdStudent_scratchStudent_dst[k] for k in sorted(kdStudent_scratchStudent_dst.keys())])
            })


        if "TSNE" in args.metrics:
            teacher_img = compute_TSNE(teacher, dataset_root=dataset_location, batch_size=args.batch_size,
                                    num_workers=args.workers, output_path="./TSNE_img/teacher_TSNE.png" )
            
            nayerStudent_img = compute_TSNE(nayerStudent, dataset_root=dataset_location, batch_size=args.batch_size,
                                    num_workers=args.workers, output_path="./TSNE_img/student_TSNE.png" )
            
            scratchStudent_img = compute_TSNE(scratchStudent, dataset_root=dataset_location, batch_size=args.batch_size, 
                                            num_workers=args.workers, output_path="./TSNE_img/scratch_stud_TSNE.png" )
            
            kdStudent_img = compute_TSNE(KDstudent, dataset_root=dataset_location, batch_size=args.batch_size,
                                    num_workers=args.workers, output_path="./TSNE_img/kdStudent_TSNE.png" )
            
            wandb.log({
                "TSNE - Teacher": wandb.Image(teacher_img),
                "TSNE - NAYER Student": wandb.Image(nayerStudent_img),
                "TSNE - Scratch Student": wandb.Image(scratchStudent_img),
                "TSNE - KD Student": wandb.Image(kdStudent_img)
            })


        if "DICE" in args.metrics:
            teacher_nayerStud_DICE = teacher_nayerStud_Comparator.dice_coefficient()
            kdStud_nayerStud_DICE = kdStud_nayerStud_Comparator.dice_coefficient()
            scratchStus_nayerStud_DICE = scratchStud_nayerStud_Comparator.dice_coefficient()
            
            wandb.log({
                "Teacher‑NayerStudent DICE score (per class)": wandb.Histogram([teacher_nayerStud_DICE[k] for k in sorted(teacher_nayerStud_DICE.keys())]),
                "KDstudent-NayerStudent DICE score (per class)": wandb.Histogram([kdStud_nayerStud_DICE[k] for k in sorted(kdStud_nayerStud_DICE.keys())]),
                "ScratchStudent-NayerStudent DICE score (per class)": wandb.Histogram([scratchStus_nayerStud_DICE[k] for k in sorted(scratchStus_nayerStud_DICE.keys())])
            })
            

        if "confusionMatrix" in args.metrics:
            compute_confusion_matrix(teacher, dataset_location, batch_size=args.batch_size, 
                            output_path=f'./Confusion_IMG/{args.teacher}confusion_matrix.png')
            compute_confusion_matrix(nayerStudent, dataset_location, batch_size=args.batch_size,
                            output_path='./Confusion_IMG/nayerStudent_confusion_matrix.png')
            compute_confusion_matrix(scratchStudent, dataset_location, batch_size=args.batch_size,
                            output_path='./Confusion_IMG/scratchStudent_confusion_matrix.png')
            compute_confusion_matrix(KDstudent, dataset_location, batch_size=args.batch_size,
                            output_path='./Confusion_IMG/KDstudent_confusion_matrix.png')

            wandb.log({
                "Confusion Matrix - Teacher": wandb.Image(f'./Confusion_IMG/{teacher}_confusion.png'),
                "Confusion Matrix - NAYER Student ": wandb.Image(f'./Confusion_IMG/{nayerStudent}_confusion.png'),
                "Confusion Matrix - Scratch Student": wandb.Image(f'./Confusion_IMG/{scratchStudent}_confusion.png'),
                "Confusion Matrix - KD Student": wandb.Image(f'./Confusion_IMG/{KDstudent}_confusion.png')
            })
        

        if "JSindex" in args.metrics:
            teacher_nayerStud_JS = teacher_nayerStud_Comparator.jensen_Shannon_index()
            kdStud_nayerStud_JS = kdStud_nayerStud_Comparator.jensen_Shannon_index()
            scratchStud_nayerStud_JS = scratchStud_nayerStud_Comparator.jensen_Shannon_index()
            
            wandb.log({
                "Jensen-Shannon Index - Teacher/Nayer Student": teacher_nayerStud_JS,
                "Jensen-Shannon Index - KDstudent/Nayer Student": kdStud_nayerStud_JS,
                "Jensen-Shannon Index - ScratchStudent/Nayer Student": scratchStud_nayerStud_JS
            })
            
        # Sincronizza automaticamente i risultati su wandb    
        result = subprocess.run([f"wandb sync {wandb.run.dir}/.."], shell=True, capture_output=True, text=True) #rimuove "/files" dal path
        print(result.stdout)



def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global time_cost
    args.gpu = gpu

    ### Integrazione da imagenet_inversion.py
    coefficients = dict()
    coefficients["r_feature"] = args.r_feature
    coefficients["first_bn_multiplier"] = args.first_bn_multiplier
    coefficients["tv_l1"] = args.tv_l1
    coefficients["tv_l2"] = args.tv_l2
    coefficients["l2"] = args.l2
    coefficients["main_loss_multiplier"] = args.main_loss_multiplier
    coefficients["adi_scale"] = args.adi_scale


    ############################################
    # GPU and FP16
    ############################################
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler = GradScaler() if args.fp16 else None
        args.autocast = autocast
    else:
        args.autocast = datafree.utils.dummy_ctx

    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-' + args.log_tag

    #### Abbreviazione Logger
    # log_name = 'R%d-%s-%s-%s%s' % (args.rank, args.dataset, args.teacher, args.student, args.log_tag) \
    #     if args.multiprocessing_distributed else '%s-%s-%s' % (args.dataset, args.teacher, args.student)
    log_name = 'R%d-%s-%s' % (args.rank, args.dataset, args.log_tag) \
        if args.multiprocessing_distributed else '%s' % (args.dataset)
    # args.logger = datafree.utils.logger.get_logger(log_name, output='checkpoints/datafree-%s/log-%s-%s-%s%s.txt'
    #                                                                 % (args.method, args.dataset, args.teacher,
    #                                                                    args.student, args.log_tag))
    args.logger = datafree.utils.logger.get_logger(log_name, output='checkpoints/datafree-%s/log-%s-%s.txt'
                                                                    % (args.method, args.dataset, args.log_tag))
    if args.rank <= 0:
        for k, v in datafree.utils.flatten_dict(vars(args)).items():  # print args
            args.logger.info("%s: %s" % (k, v))

    name_project = 'datafree-%s/log-%s-%s-%s%s.txt' % (
    args.method, args.dataset, args.teacher, args.student, args.log_tag)
    wandb.init(project="invNAYER",
               name=name_project,
               tags="t1",
               config=args.__dict__,
               mode=args.wandb)

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    print("get_dataset")
    args.num_classes = num_classes
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)

    ############################################
    # Setup models
    ############################################
    def prepare_model(model):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
            return model
        elif args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                return model
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
                return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model

    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()

    if args.train_distilled_student:
        nayerStudent = registry.get_model(args.student, num_classes=num_classes, pretrained=True).eval()
        nayerStudent.load_state_dict(torch.load(args.nayer_student, map_location='cpu')['state_dict'])
        nayerStudent = prepare_model(nayerStudent)


    args.normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    # pretrain = torch.load('checkpoints/pretrained/%s_%s.pth' % (args.dataset, args.teacher),
    #                                    map_location='cpu')['state_dict']
    # FIXME: path without ".."
    if args.dataset != 'imagenet':
        teacher.load_state_dict(torch.load('./checkpoints/pretrained/%s_%s.pth' % (args.dataset, args.teacher),
                                           map_location='cpu')['state_dict'])
    student = prepare_model(student)
    teacher = prepare_model(teacher)

    '''
        Salta la sintetizzazione di immagini nel caso in cui 
        si vogliano calcolare solo le metriche
    '''
    if args.metrics: 
        return teacher, student, None, None
    
    criterion = datafree.criterions.KLDiv(T=args.T)

    ############################################
    # Setup the data-free synthesizer
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size

    if args.method == 'nayer':
        le_name = "label_embedding/" + args.dataset + "_le.pickle"
        with open(le_name, "rb") as label_file:
            label_emb = pickle.load(label_file)
            label_emb = label_emb.to(args.gpu).float()

        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            generator = datafree.models.generator.NLGenerator(ngf=64, img_size=32, nc=3, nl=num_classes,
                                                             label_emb=label_emb, le_emb_size=args.le_emb_size,
                                                             sbz=args.synthesis_batch_size)
            generator = prepare_model(generator)
            synthesizer = datafree.synthesis.NAYER(teacher, student, generator,
                                                  num_classes=num_classes, img_size=(3, 32, 32), init_dataset=args.cmi_init,
                                                  save_dir=args.save_dir, device=args.gpu, transform=ori_dataset.transform,
                                                  normalizer=args.normalizer, num_workers=args.workers,
                                                  synthesis_batch_size=args.synthesis_batch_size,
                                                  sample_batch_size=args.batch_size,
                                                  g_steps=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, adv=args.adv,
                                                  bn=args.bn, oh=args.oh, bn_mmt=args.bn_mmt, contr=args.contr,
                                                  g_life=args.g_life, g_loops=args.g_loops, gwp_loops=args.gwp_loops,
                                                  coefficients=coefficients)
        elif args.dataset == 'tiny_imagenet':
            generator = datafree.models.generator.NLGenerator(ngf=64, img_size=64, nc=3, nl=num_classes,
                                                             label_emb=label_emb, le_emb_size=args.le_emb_size,
                                                             sbz=args.synthesis_batch_size)
            generator = prepare_model(generator)
            synthesizer = datafree.synthesis.NAYER(teacher, student, generator,
                                                  num_classes=num_classes, img_size=(3, 64, 64), init_dataset=args.cmi_init,
                                                  save_dir=args.save_dir, device=args.gpu, transform=ori_dataset.transform,
                                                  normalizer=args.normalizer,
                                                  synthesis_batch_size=args.synthesis_batch_size,
                                                  sample_batch_size=args.batch_size,
                                                  g_steps=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, adv=args.adv,
                                                  bn=args.bn, oh=args.oh, bn_mmt=args.bn_mmt,
                                                  g_life=args.g_life, g_loops=args.g_loops, gwp_loops=args.gwp_loops)
        elif args.dataset == 'imagenet':
            generator = datafree.models.generator.NLDeepGenerator(ngf=64, img_size=224, nc=3, nl=num_classes,
                                                                  label_emb=label_emb, le_emb_size=args.le_emb_size,
                                                                  sbz=args.synthesis_batch_size)
            generator = prepare_model(generator)
            synthesizer = datafree.synthesis.NAYER(teacher, student, generator,
                                                  num_classes=num_classes, img_size=(3, 244, 244), init_dataset=args.cmi_init,
                                                  save_dir=args.save_dir, device=args.gpu, transform=ori_dataset.transform,
                                                  normalizer=args.normalizer,
                                                  synthesis_batch_size=args.synthesis_batch_size,
                                                  sample_batch_size=args.batch_size,
                                                  g_steps=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, adv=args.adv,
                                                  bn=args.bn, oh=args.oh, bn_mmt=args.bn_mmt, dataset=args.dataset,
                                                  g_life=args.g_life, g_loops=args.g_loops, gwp_loops=args.gwp_loops)
    else:
        raise NotImplementedError

    ############################################
    # Setup optimizer
    ############################################
    optimizer = torch.optim.SGD(student.parameters(), args.lr, weight_decay=args.weight_decay,
                                momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int((args.epochs - args.warmup)), eta_min=args.eta_min)

    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if isinstance(student, nn.Module):
                student.load_state_dict(checkpoint['state_dict'])
            else:
                student.module.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try:
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                print("Fails to load additional model information")
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))

    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc']))
        return

    ############################################
    # Train Loop
    ############################################
    if args.train_distilled_student:
        '''
            Da:  train_scratch.py
            Genera un train_loader o dataLoader per lo studente da distillare
            Elimina la necessità di un generatore come da workflow NAYER base.
        '''
        device = args.gpu if args.gpu is not None else 'cpu'
        num_classes, train_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
        cudnn.benchmark = True
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        dataLoader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        evaluator = datafree.evaluators.classification_evaluator(val_loader)

        distilledStudent = registry.get_model(args.student, num_classes=num_classes).to(device)


    for epoch in range(args.start_epoch, args.epochs):
        tm = time.time()
        args.current_epoch = epoch

        ''' SKIP durante addestramento KD STUDENT'''   
        if not args.train_distilled_student:
            vis_results, cost, loss_synthesizer, loss_oh, loss_var_l1, loss_var_l2, loss_l2, loss_bn, loss_adv = synthesizer.synthesize()  # g_steps
        # if vis_results is not None:
        #     for vis_name, vis_image in vis_results.items():
        #         str_epoch = str(epoch)
        #         if epoch < 100:
        #             str_epoch = "0" + str_epoch
        #         if epoch < 10:
        #             str_epoch = "0" + str_epoch
        #         for save_iter in range(len(vis_image)):
        #             datafree.utils.save_image_batch(vis_image[save_iter], 'checkpoints/datafree-%s/%s%s%s-%s.png'
        #                                             % (args.method, vis_name, args.log_tag, str_epoch, save_iter))
            time_cost += cost #SKIP durante addestramento KD STUDENT
        ''' Adattamento per riutilizzabilità codice NAYER esistente ''' 
        vis_results, cost, loss_synthesizer, loss_oh, loss_var_l1, loss_var_l2, loss_l2, loss_bn, loss_adv = 0, 0, 0, 0, 0, 0, 0, 0, 0
            
        if epoch >= args.warmup:
            del vis_results
            # del vis_image
            # del vis_name
            ''' Modificato con dataLoader al posto del modulo synthesizer '''
            if args.train_distilled_student:
                train_distilled_student(dataLoader, teacher, nayerStudent, distilledStudent, optimizer, args)
            else:
                #NAYER default function
                train(synthesizer, [student, teacher], criterion, optimizer, args)  # kd_steps
        tm = time.time() - tm


        student.eval()
        if epoch >= args.warmup:
            eval_results = evaluator(student, device=args.gpu)
            (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        else:
            acc1 = acc5 = val_loss = 0

        args.logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Lr={lr:.4f} '
                         'SL:{sl:.4f} OH:{oh:.4f} Cost:{co:.4f} Time:{tm:.4f}'.format(current_epoch=args.current_epoch, acc1=acc1,
                                                                        acc5=acc5, loss=val_loss,
                                                                        lr=optimizer.param_groups[0]['lr'],
                                                                        sl=loss_synthesizer, oh=loss_oh,
                                                                        co=cost, tm=tm))
        wandb.log({"Acc1": acc1, "Acc5": acc5, "VLoss": val_loss, "lr": optimizer.param_groups[0]['lr'],
                    "SLoss": loss_synthesizer, "OHLoss": loss_oh, \
                    "loss_var_l1":loss_var_l1, "loss_var_l2":loss_var_l2, "loss_l2":loss_l2, \
                    "loss_BN":loss_bn, "loss_ADV": loss_adv })

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = 'checkpoints/datafree-%s/%s-%s-%s-%s.pth' % (args.method, args.dataset, args.teacher, args.student,
                                                                  args.log_tag)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, _best_ckpt)

        if epoch >= args.warmup:
            scheduler.step()

    if args.rank <= 0:
        args.logger.info("Best: %.4f" % best_acc1)
        args.logger.info("Generation Cost: %1.3f" % (time_cost / 3600.))
    return teacher, student, synthesizer, evaluator


# do the distillation
def train(synthesizer, model, criterion, optimizer, args, dataLoader = None):
    global time_cost
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1, 5))
    student, teacher = model
    optimizer = optimizer
    student.train()
    teacher.eval()
    for i in range(args.kd_steps):
        if args.method in ['zskt', 'dfad', 'dfq', 'dafl']:
            images, cost = synthesizer.sample()
            time_cost += cost
        else:
            images = synthesizer.sample()
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        with args.autocast():
            with torch.no_grad():
                t_out, t_feat = teacher(images, return_features=True)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())

            #distillazione "classica" con addestramento studente con gli stessi dati dell'insegnante
            #new_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
            #loss_crossE = new_criterion(s_out, target)
            #main_loss = "alpha"*loss_crossE + "1-alpha"*loss_s => magari sovrascrivere ni modo da non modificare tutto il resto
            
        optimizer.zero_grad()
        if args.fp16:
            scaler_s = args.scaler_s
            scaler_s.scale(loss_s).backward()
            scaler_s.step(optimizer)
            scaler_s.update()
        else:
            loss_s.backward()
            optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        if args.print_freq == -1 and i % 10 == 0 and args.current_epoch >= 150:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info(
                '[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, '
                'train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
                .format(current_epoch=args.current_epoch, i=i, total_iters=args.kd_steps, train_acc1=train_acc1,
                        train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()
        elif args.print_freq > 0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info('[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, '
                             'train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
                             .format(current_epoch=args.current_epoch, i=i, total_iters=args.kd_steps,
                                     train_acc1=train_acc1,
                                     train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()


                           # ↓ da train_scratch. Equivale a dataLoader
def train_distilled_student(train_loader, teacher, nayerStudent, distilledStudent, optimizer, args):
    import torch.nn.functional as F
    global best_acc1
    loss_metric = datafree.metrics.RunningLoss(nn.CrossEntropyLoss(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1, 5))
    #device = args.gpu if args.gpu is not None else 'cpu'

    distilledStudent.train()
    teacher.eval()
    nayerStudent.eval()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

    for i, (images, targets) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            targets = targets.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()
        with args.autocast(enabled=args.fp16):
            with torch.no_grad():
                teacher_logits = teacher(images)
                nayer_logits = nayerStudent(images)
            student_logits = distilledStudent(images)
            # BCE tra output studente e target del teacher (one-hot)
            teacher_targets = F.one_hot(targets, num_classes=teacher_logits.shape[1]).float()
            loss_bce = bce_loss(student_logits, teacher_targets)
            # KL tra output studente e output nayerStudent (softmax)
            loss_kl = kl_loss(
                F.log_softmax(student_logits, dim=1),
                F.softmax(nayer_logits, dim=1)
            )
            loss = args.alpha * loss_bce + (1 - args.alpha) * loss_kl
            #*****loss_crossE**** = criterion(output, target)

        # measure accuracy and record loss
        acc_metric.update(student_logits, targets)
        loss_metric.update(student_logits, targets)

        if args.fp16:
            scaler = args.scaler if hasattr(args, 'scaler') else None
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            loss.backward()
            optimizer.step()

        if args.print_freq > 0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info('[kdTrain] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, '
                             'train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
                             .format(current_epoch=args.current_epoch, i=i, total_iters=len(train_loader),
                                     train_acc1=train_acc1, train_acc5=train_acc5, train_loss=train_loss,
                                     lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


if __name__ == '__main__':
    main()
