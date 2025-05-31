def sideBy_barplot(save_path, *data_lists, labels=None, title="Barplot", xlabel="X", ylabel="Y", xticks=[], width=0.8):
	"""
	Salva su barplot affiancato di più liste in ingresso.
	
	Args:
		save_path (str): Percorso dove salvare il file .png
		*data_lists: Liste di dati da plottare
		labels (list): Lista di etichette per la legenda
		title: Titolo del grafico
		xlabel
		ylabel
		colors (list): Lista di colori per le barre
		width (float): Larghezza delle barre
	Returns:
		Path di salvataggio
	"""
	import matplotlib.pyplot as plt
	import numpy as np
	import os
	os.makedirs(os.path.dirname(save_path), exist_ok=True)

	fig, ax = plt.subplots(figsize=(10, 6))
	
	n_lists = len(data_lists)
	x = np.arange(len(data_lists[0]))
	width = width / n_lists  # Larghezza effettiva per ogni barra
	colors = plt.cm.tab10(np.linspace(0, 1, n_lists))
	
	if labels is None:
		labels = [f'Lista {i+1}' for i in range(n_lists)]

	for i, data in enumerate(data_lists):
		offset = width * i - (width * (n_lists-1))/2
		bars = ax.bar(x + offset, data, width, label=labels[i], color=colors[i])
		
		# Aggiunge valori sopra le barre
		for bar in bars:
			height = bar.get_height()
			ax.text(bar.get_x() + bar.get_width()/2., height,
				   f'{height:.2f}', ha='center', va='bottom')

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.set_xticks(x, labels=xticks)
	ax.legend(loc='best')

	plt.tight_layout()
	plt.savefig(save_path, bbox_inches='tight')
	plt.close()
	
	return save_path