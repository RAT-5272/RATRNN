from Config import Config
from Config import StageConfig
from Config import Stage1

import os
import re
import random

# Data Storage
import json

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
	if Folders == []:
		raise ValueError("ReadFiles cannot take in an empty list of folders")
	
	for i in range(len(Folders)):
		Folders[i] = os.path.expanduser(Folders[i])
		Folders[i] = os.path.join(Config.TRAINING_DATA_PATH, Folders[i])

	if LoadCachedData:
		print("Attempting to load cached training data to skip processing...")
		try:
			with open(f"{Config.DATASET_CACHE_PATH}/Dataset.json", "r") as File:
				MessageHistories = json.load(File)
				print(Style.BRIGHT + Fore.MAGENTA + "  Loaded dataset!")
				return MessageHistories
		except Exception as e:
			print(f"  Failed to load dataset, instead building from new, {e}")
	
	print(Style.BRIGHT + Fore.MAGENTA + "Parsing training data")
	
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
	print("  Extracting contents of folders:")
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

	print("  Parsing messages from files")
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


	print("  Saving database")
	try:
		with open(f"{Config.DATASET_CACHE_PATH}/Dataset.json", "w+") as File:
			json.dump(MessageHistories, File, indent=4)
	except Exception as e:
		print(f"    Failed to save database, {e}")

		
	print(Style.BRIGHT + Fore.MAGENTA + f"Extracted {TotalMessageCount} messages from {FileCount} files")
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

def ParseTrainingData(Messages: dict, TrainingConfig: StageConfig, LoadCachedData: bool = True):
	
	AllUserMessages = []
	MyMessages = []
	FineTuningMessages = []


	TotalMessageCount = 0
	for File in Messages:
		TotalMessageCount += len(File["Messages"])
	ProgressBar = tqdm(total = TotalMessageCount, desc="  ", unit = " messages", colour = "#b5a3c2")
	
	for File in Messages:
		Filepath = File["FilePath"]
		FileMessageCount = len(File["Messages"])
		for Index in range(max(FileMessageCount - TrainingConfig.MESSAGE_HISTORY_LENGTH - 1, 1)):
			ProgressBar.update(1)

			ConversationHistory = File["Messages"][Index:Index + TrainingConfig.MESSAGE_HISTORY_LENGTH + 1].copy()
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
				if random.random() < GetHistoryProbability(i, len(ProcessedMessageHistories), TrainingConfig.MAX_LENGTH_BIAS):
					ProcessedMessageHistories.append(ProcessedMessageHistory[-(i+1):].copy())
			
			# Classify then add the message histories to the correct type
			if "FineTuning" in Filepath:
				FineTuningMessages.append("")
			elif PredictSenderUsername == "rat_5272":
				MyMessages.append("")
			elif PredictSenderUsername != "rat_5272":
				AllUserMessages.append("")
				

	print(len(FineTuningMessages))
	print(len(AllUserMessages))
	print(len(MyMessages))

	TrainingData = []

	# Absolute size
	for MessageHistory in FineTuningMessages:
		if random.random() < TrainingConfig.DATASET_SCALE and random.random < TrainingConfig.ABSOLUTE_DATASET_CONTENTS["FineTuning"]:
			TrainingData.append(MessageHistory)
	
	ProgressBar.close()

# MUST MUST MUS MUST ALSO ADD IT TO DO THIS:
# DELETE PREVIOUS TOOL INSTANCES BEFORE FINAL MESSAGE YK YK