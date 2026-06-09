from Config import Config
from Config import StageConfig
from Config import Stage1

import os
import re
import random

# Data Storage
import json
import hashlib

# Logging
from colorama import Fore, Back, Style
from colorama import init as InitColorama
InitColorama(autoreset=True)
from tqdm import tqdm



#print(Style.BRIGHT + Fore.MAGENTA + "This text is red")
#print(Back.GREEN + "This has a green background")
#print(Style.BRIGHT + "This is bright text")



def ReadFiles(Folders: list[str] = [""], ValidExtensions: tuple[str] = (".txt"), LoadCachedData: bool = True):
	"""
	||| DOES NOT CONSTRUCT A TRAINABLE DATASET |||
	
	This function will return a dictionary of files and their contents
	"""

	print(Style.BRIGHT + Fore.MAGENTA + "Reading raw conversation logs")


	if Folders == []:
		raise ValueError("ReadFiles cannot take in an empty list of folders")
	
	for i in range(len(Folders)):
		Folders[i] = os.path.expanduser(Folders[i])
		Folders[i] = os.path.join(Config.TRAINING_DATA_PATH, Folders[i])

	if LoadCachedData:
		print("\n  Attempting to load cached message data to skip processing...")
		try:
			with open(f"{Config.DATASET_CACHE_PATH}/RawMessages.json", "r") as File:
				MessageHistories = json.load(File)

				TotalMessageCount = 0
				for File in MessageHistories:
					TotalMessageCount += len(File["Messages"])

				print(Style.BRIGHT + Fore.MAGENTA + f"    Loaded {TotalMessageCount} messages from the cache")
				return MessageHistories
		except Exception as e:
			print(f"    Failed to load messages, instead building from new, {e}")
	
	for i in range(len(Folders)):
		Folders[i] = os.path.join(Config.TRAINING_DATA_PATH, Folders[i])

	# Count files
	TotalFileCount = 0
	for Folder in Folders:
		for Root, _, Files in os.walk(Folder):
			for F in Files:
				if F.endswith(ValidExtensions):
					TotalFileCount += 1

	FileCount = 0
	print("\n  Extracting contents of folders:")
	ProgressBar = tqdm(total = TotalFileCount, desc="  ", unit = " folders", colour= "#b5a3c2")

	# Open all training files
	TrainingData = []
	for Folder in Folders:
		for Root, Dirs, Files in os.walk(Folder):
			for File in Files:
				if not File.endswith(ValidExtensions):
					continue

				Path = os.path.join(Root, File)
				with open(Path, "r", encoding="utf-8") as TrainingFile:
					Text = TrainingFile.read()
					FileCount += 1
					TrainingData.append([Text, Path])
				ProgressBar.update(1)
	ProgressBar.close()

	# Count messages
	TotalMessageCount = 0
	for Content, Path in TrainingData:
		TotalMessageCount += Content.count("<BOT_MsgHeadStart>")

	print("\n  Parsing messages from files")
	ProgressBar = tqdm(total = TotalMessageCount, desc="  ", unit = " messages", colour = "#b5a3c2")

	"""Read message histories"""
	HeadPattern = re.compile(r"<BOT_MsgHeadStart>(.*?)<BOT_MsgHeadEnd>", re.S)
	ReplyPattern = re.compile(r"<BOT_MsgReplyStart>(.*?)<BOT_MsgReplyEnd>", re.S)
	ContentPattern = re.compile(r"<BOT_MsgContentStart>(.*?)<BOT_MsgContentEnd>", re.S)

	MessageHistories = []

	for Content, FilePath in TrainingData:

		Heads = HeadPattern.findall(Content)
		Replies = ReplyPattern.findall(Content)
		Messages = ContentPattern.findall(Content)

		FileHistory = []
		MessageIdUserLookup = {}
		LastMessageTime = None

		for Head, Reply, Message in zip(Heads, Replies, Messages):
			#print(Head, Reply, Message)

			ProgressBar.update(1)

			Parts = Head.split()

			if len(Parts) < 4:
				continue

			MessageId = Parts[0]
			Username = Parts[2]

			DisplayName = " ".join(Parts[3:-2]) or "NONE"

			try:
				MessageTime = int(Parts[-2])
			except ValueError:
				MessageTime = 0

			if LastMessageTime is None:
				TimeGap = 0
			else:
				TimeGap = MessageTime - LastMessageTime

			LastMessageTime = MessageTime

			ReplyId = Parts[-1]

			if ReplyId == "0":
				ReplyUser = MessageIdUserLookup.get(ReplyId, "NONE")
			else:
				ReplyUser = "NONE"

			MessageIdUserLookup[MessageId] = Username

			FileHistory.append({
				"Username": Username,
				"DisplayName": DisplayName,
				"TimeGap": TimeGap,
				"ReplyUser": ReplyUser,
				"Reply": Reply,
				"Message": Message
			})

		MessageHistories.append({
			"FilePath": FilePath,
			"Messages": FileHistory
		})
		
	ProgressBar.close()

	if MessageHistories == []:
		raise ValueError("MessageHistories is an empty list, perhaps the filepaths used are incorrect?")


	print("\n  Saving messages...")
	try:
		with open(f"{Config.DATASET_CACHE_PATH}/RawMessages.json", "w+") as File:
			json.dump(MessageHistories, File, indent=4)
	except Exception as e:
		print(f"    Failed to save messages, {e}")
	print("  Saved all messages")

		
	print(Style.BRIGHT + Fore.MAGENTA + f"\nExtracted {TotalMessageCount} messages from {FileCount} files")
	return MessageHistories



def GetHistoryProbability(i, MessageHistoryLength, MAX_LENGTH_BIAS):
    if MessageHistoryLength <= 1:
        return 1.0

    # Uniform case
    if MAX_LENGTH_BIAS <= 0:
        return 1.0 / MessageHistoryLength

    # Fully biased case
    if MAX_LENGTH_BIAS >= 1:
        return 1.0 if i == MessageHistoryLength - 1 else 0.0

    # Convert bias into an exponent
    Exponent = MAX_LENGTH_BIAS / (1 - MAX_LENGTH_BIAS)

    # Weight for this history length
    Weight = ((i + 1) / MessageHistoryLength) ** Exponent

    # Normalize
    TotalWeight = sum(
        ((j + 1) / MessageHistoryLength) ** Exponent
        for j in range(MessageHistoryLength)
    )

    return Weight / TotalWeight

def CreateTrainingSamples(Messages: dict, TrainingConfig: StageConfig, LoadCachedData: bool = True):
	print(Style.BRIGHT + Fore.MAGENTA + f"Creating training data for stage '{TrainingConfig.NAME}'")


	if LoadCachedData:
		print("\n  Attempting to load cached samples to skip processing...")
		try:
			with open(f"{Config.DATASET_CACHE_PATH}/TrainingSamples.json", "r") as File:
				TrainingData = json.load(File)
				print(Style.BRIGHT + Fore.MAGENTA + f"    Loaded {len(TrainingData)} samples from the cache")
				return TrainingData
		except Exception as e:
			print(f"    Failed to load samples, instead building from new, {e}")



	
	AllUserMessages = []
	MyMessages = []
	FineTuningMessages = []


	TotalMessageCount = 0
	for File in Messages:
		TotalMessageCount += max(len(File["Messages"])-1, 1)
	
	print("\n  Constructing conversations")
	ProgressBar = tqdm(total = TotalMessageCount, desc="  ", unit = " messages", colour = "#b5a3c2")
	
	for File in Messages:
		Filepath = File["FilePath"]
		FileMessageCount = len(File["Messages"])
		for Index in range(max(FileMessageCount - 1, 1)):
			ProgressBar.update(1)

			Start = max(0, Index - TrainingConfig.MESSAGE_HISTORY_LENGTH + 1)

			ConversationHistory = File["Messages"][Start:Index + 2].copy()
			MessageToPredict = ConversationHistory[-1].copy()
			ConversationHistory = ConversationHistory[:-1]

			PredictSenderUsername = MessageToPredict["Username"]

			# Don't train on system prompts / tool responses
			if PredictSenderUsername == "RATRNN_system" or PredictSenderUsername == "RATRNN_tool":
				continue

			# Convert from data heavy format to LLM friendly
			ProcessedMessageHistory = []
			for Message in ConversationHistory:
				if Message["Username"] == "RATRNN_tool":
					ProcessedMessageHistory.append({
						"role": "tool",
						"content": Message["Message"]
					})
				elif Message["Username"] == "RATRNN_system":
					ProcessedMessageHistory.append({
						"role": "system",
						"content": Message["Message"]
					})
				else:
					if Message["Username"] == PredictSenderUsername:
						ProcessedMessageHistory.append({
							"role": "assistant",
							"content": Message["Message"]
						})
					else:
						ProcessedMessageHistory.append({
							"role": "user",
							"content": f"{Message['DisplayName']}\n{Message['Message']}"
						})
			


			
			
			# Add a message history of each length to ProcessedMessageHistories
			ProcessedMessageHistories = []
			for i in range(len(ProcessedMessageHistory)):
				if random.random() < GetHistoryProbability(i, len(ProcessedMessageHistory), TrainingConfig.MAX_LENGTH_BIAS) * TrainingConfig.TARGET_SAMPLES_PER_MESSAGE:
					ProcessedMessageHistories.append(ProcessedMessageHistory[-(i+1):].copy())
			

			# Classify then add the message histories to the correct type
			AssistantMessage = {
				"role": "assistant",
				"content": MessageToPredict["Message"]
			}

			for History in ProcessedMessageHistories:
				Sample = {
					"Source": "FineTuning" if "FineTuning" in Filepath else
							  "MyMessages" if PredictSenderUsername == "rat_5272" else
							  "AllUserMessages",
					
					"Messages": History + [AssistantMessage]
				}

				if "FineTuning" in Filepath:
					FineTuningMessages.append(Sample)
				elif PredictSenderUsername == "rat_5272":
					MyMessages.append(Sample)
				elif PredictSenderUsername != "rat_5272":
					AllUserMessages.append(Sample)

	ProgressBar.close()

	TrainingData = []

	print("\n  Creating final dataset (Absolute and Relative size)")

	# Absolute size
	ProgressBar = tqdm(total = len(FineTuningMessages) + len(MyMessages) + len(AllUserMessages), desc="  ", unit = " samples", colour = "#b5a3c2")
	for MessageHistory in FineTuningMessages:
		ProgressBar.update(1)
		if (random.random() < TrainingConfig.DATASET_SCALE and random.random() < TrainingConfig.ABSOLUTE_DATASET_CONTENTS["FineTuning"]) or TrainingConfig.FULL_FINE_TUNING_DATASET:
			TrainingData.append(MessageHistory)
	
	for MessageHistory in MyMessages:
		ProgressBar.update(1)
		if random.random() < TrainingConfig.DATASET_SCALE and random.random() < TrainingConfig.ABSOLUTE_DATASET_CONTENTS["MyMessages"]:
			TrainingData.append(MessageHistory)
	
	for MessageHistory in AllUserMessages:
		ProgressBar.update(1)
		if random.random() < TrainingConfig.DATASET_SCALE and random.random() < TrainingConfig.ABSOLUTE_DATASET_CONTENTS["AllUserMessages"]:
			TrainingData.append(MessageHistory)
	ProgressBar.close()

	# Relative size
	RelativeSets = {
		"AllUserMessages": AllUserMessages,
		"MyMessages": MyMessages,
		"FineTuning": FineTuningMessages
	}

	NonZero = []

	for Name, Weight in TrainingConfig.RELATIVE_DATASET_CONTENTS.items():
		if Weight > 0:
			NonZero.append(len(RelativeSets[Name]) / Weight)
	
	try:
		BaseSize = min(NonZero)
	except:
		BaseSize = 0

	ProgressBar = tqdm(total = 3, desc="  ", unit = " datasets", colour = "#b5a3c2")
	for Name, Weight in TrainingConfig.RELATIVE_DATASET_CONTENTS.items():
		ProgressBar.update(1)
		if Weight <= 0: continue

		TargetSize = int(BaseSize * Weight * TrainingConfig.DATASET_SCALE)

		Dataset = RelativeSets[Name]

		if len(Dataset) <= TargetSize:
			TrainingData.extend(Dataset)
		else:
			TrainingData.extend(random.sample(Dataset, TargetSize))
	ProgressBar.close()


	print("\n  Deduplicating samples")
	def MakeKey(Messages):
		return hashlib.sha1(
			json.dumps(Messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
		).hexdigest()
	
	
	Seen = set()
	DeduplicatedTrainingData = []

	for Sample in TrainingData:
		Source = Sample["Source"]
		Messages = Sample["Messages"]

		if Source == "FineTuning" and TrainingConfig.FULL_FINE_TUNING_DATASET:
			DeduplicatedTrainingData.append(Messages)
			continue

		Key = MakeKey(Messages)

		if Key not in Seen:
			Seen.add(Key)
			DeduplicatedTrainingData.append(Messages)

	TrainingData = DeduplicatedTrainingData

	print("\n  Saving samples...")
	try:
		with open(f"{Config.DATASET_CACHE_PATH}/TrainingSamples.json", "w+") as File:
			json.dump(TrainingData, File, indent=4)
	except Exception as e:
		print(f"    Failed to save samples, {e}")
	print("  Saved all samples")

	print(Style.BRIGHT + Fore.MAGENTA + f"\nCreated {len(TrainingData)} training samples")

	return TrainingData
	
	

# MUST MUST MUS MUST ALSO ADD IT TO DO THIS:
# DELETE PREVIOUS TOOL INSTANCES BEFORE FINAL MESSAGE YK YK