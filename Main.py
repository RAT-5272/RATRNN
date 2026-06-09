def Main():
	import TrainingData
	import Config

	MessageData = TrainingData.ReadFiles()
	print("\n\n\n")
	Stage1Messages = TrainingData.CreateTrainingSamples(MessageData, Config.Stage1)


	

if __name__ == "__main__":
	Main()