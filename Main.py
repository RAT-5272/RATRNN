def Main():
	import TrainingData
	import Config

	MessageData = TrainingData.ReadFiles()
	Stage1Messages = TrainingData.ParseTrainingData(MessageData, Config.Stage1)

	print(Stage1Messages)

	

if __name__ == "__main__":
	Main()