

def TrainingLoop():
	import TrainingData.TrainingDataParser as TrainingDataParser
	import Config
	from Model import Train, LoadModel
	from TrainingData.DatasetBuilder import BuildDataset, PrintDatasetStats

	TRAINING_STAGES = [Config.Stage1, Config.Stage2, Config.Stage3]

	Model, Tokenizer = LoadModel(BaseModel = True)

	# Read all messages from files
	MessageData = TrainingDataParser.ReadFiles()
	print("\n\n\n")

	# Training loop
	for StageConfig in TRAINING_STAGES:

		# Format all the messages into conversations (samples)
		Samples = TrainingDataParser.CreateTrainingSamples(MessageData, StageConfig, BotUsername = "binguslord8060", )

		# Tokenize all samples, force system prompt to top, apply custom token weights, etc..
		Dataset = BuildDataset(Samples, Tokenizer)
		#PrintDatasetStats(Dataset)
		del Samples

		# Run the training for this stage, and then save when it's done
		Model = Train(Model, Tokenizer, Dataset, StageConfig)
		del Dataset

def InferenceLoop():
	from Inference.Frontend import InferenceManager
	import Config
	
	ChatLoop = InferenceManager(Config.Config, Config.InferenceConfig)

	ConversationID = ChatLoop.CreateConversation()






def Main():
	print("\n"*50)

	# All imports
	while True:
		print("""
Welcome to RAT RNN!
  1 - Train
  2 - Chat""")

		Choice = input()

		if Choice in ["1", "2"]:
			break
	
	match Choice:
		case "1":
			TrainingLoop()

		case "2":
			InferenceLoop()
	
	# TODO - Make this printing prettier

	


	

if __name__ == "__main__":
	Main()