import math
import os # To get the width of the terminal
from tqdm import tqdm

from datasets import Dataset

def BuildDataset(RawData, Tokenizer):
	Formatted = []

	ProgressBar = tqdm(total = len(RawData), desc="  ", unit = " samples", colour = "#b5a3c2")

	for Sample in RawData:
		ProgressBar.update(1)
		Formatted.append({
			"text": 
			Tokenizer.apply_chat_template(
				Sample,
				tokenize=False,
				add_generation_prompt=False
			)
		})

		import random
	
	ProgressBar.close()

	random.shuffle(Formatted)
	
	# TODO - Make dataset builder save to cache, pretty printing

	return Dataset.from_list(Formatted)


def PrintDatasetStats(Dataset, Columns = 10, Rows = 20):
	TerminalWidth = os.get_terminal_size().columns
	Columns = min(Columns, TerminalWidth/3)
	
	# Simple stats
	Max = 0
	Total = 0
	for Sample in Dataset:
		Length = len(Sample)
		
		Max = max(Max, Length)
		Total += Length
	
	Average = Total / len(Dataset)

	# Bucketing
	Buckets = []
	BucketSize = max(int(Max/Columns), 1)
	for i in range(BucketSize):
		Buckets.append(0)
	
	for Sample in Dataset: # Find amount of samples in each bucket size
		BucketIndex = math.ceil(len(Sample) / BucketSize)
		Buckets[BucketIndex] += 1
	
	for i in range(len(Buckets)): # Normalise buckets to row count
		Buckets[i] = (Buckets[i] / max(Buckets)) * Rows
	
	for i in range(len(Buckets)):
		BucketMax = (Buckets[i]+1) * BucketSize
	
	# TODO - FINISH LATER


