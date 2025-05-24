from skimage.metrics import structural_similarity as ssim

def compute_SSIM(image1, image2):
	"""
	Calcola l'indice di similarità strutturale (SSIM) tra due immagini.
	"""
	# Cast a float e normalizzazione tra 0 e 1
	image1 = image1.astype(float) / 255.0
	image2 = image2.astype(float) / 255.0

	ssim_index, _ = ssim(image1, image2, full=True, multichannel=True)

	return ssim_index