from unsloth import FastLanguageModel

import time
import torch
import os

import Config as Configuration

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True




class ConversationItem():
	def __init__(Self, ConversationID: int):
		Self.ConversationID = ConversationID
		Self.MessageHistory = []
		Self.KVcache = None
		Self.ChatTokens = None
		Self.SystemPromptTokens = None

class GenerationQueueItem():
	def __init__(Self, Conversation: ConversationItem, Priority: int):
		Self.Conversation = Conversation
		Self.Priority: int = Priority
		Self.EntryTime: float = time.time()

class InferenceManager():
	"""
	
	CreateConversation
	
	QueueRequest"""


	def __init__(Self, Config: Configuration.Config, InferenceConfig: Configuration.InferenceConfig):
		Self.Config: Configuration.Config = Config
		Self.InferenceConfig: Configuration.InferenceConfig = InferenceConfig

		Self.GenerationQueue: list[GenerationQueueItem] = []
		Self.NextConversationID: int = 0



		print("Loading model...")
		StartTime = time.time()

		Model, Tokenizer = FastLanguageModel.from_pretrained(
			model_name=os.path.join(Self.Config.MODEL_SAVE_PATH, "Me Only"),

			max_seq_length=Self.InferenceConfig.MAX_SEQ_LENGTH,

			dtype=torch.bfloat16,

			load_in_4bit=Self.InferenceConfig.LOAD_IN_4BIT,
			load_in_8bit=Self.InferenceConfig.LOAD_IN_8BIT,

			#attn_implementation="flash_attention_2", TURN ON FLASH ATTENTION WHEN I INSTALL IT pijgwrpi hrgjwIWRGH )ORGWH WOUGHRGW OhwrgO{ HWRG)IO hwrgiop'hj grwop'ihwrg iphrgW pihgRW iphwrg iopwrghwruiogWRGUIOWRGO HWROGHwrgOUHwrgo}
		)

		print(torch.cuda.get_device_name())
		print(Model.config._attn_implementation)

		FastLanguageModel.for_inference(Model)

		Elapsed = time.time() - StartTime

		print(f"Loaded in {Elapsed:.2f}s")
		print()

		Self.Model = Model
		Self.Tokenizer = Tokenizer
	

	def CreateConversation(Self):
		"""Creates a conversation and returns it's ID"""

		ConversationID = Self.NextConversationID

		Self.NextConversationID += 1

		return ConversationID
	
	def QueueRequest(Self):
		"""Adds a generation request to the priority queue, returns the generated output from the AI as a text streamer"""

		
		pass

	
		
