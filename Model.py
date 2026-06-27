import os
import torch
import Config

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

# Pretty logging
import time

# I don't even know if this is how you are meant to do it, all I know is that it stopped the bug ¯\_(ツ)_/¯
CFG = Config.Config()
TCFG = Config.TrainingConfig()




CurrentStageName = ""


def LoadModel(BaseModel = False, Checkpoint = False, FinishedModel = False):

	if BaseModel:
		# Get base model and tokenizer
		Model, Tokenizer = FastLanguageModel.from_pretrained(
			model_name = CFG.BASE_MODEL,
			
			max_seq_length = TCFG.MAX_SEQ_LENGTH,
			
			dtype = torch.bfloat16 if TCFG.USE_BF16 else torch.float16,
			
			load_in_4bit = CFG.LOAD_IN_4BIT,
			load_in_8bit = CFG.LOAD_IN_8BIT,

			attn_implementation="flash_attention_2",
		)

		# Load LoRA
		Model = FastLanguageModel.get_peft_model(
			Model,
			
			r = TCFG.LORA_R,
			lora_alpha = TCFG.LORA_ALPHA,
			lora_dropout = TCFG.LORA_DROPOUT,
			target_modules = TCFG.LORA_TARGET_MODULES,
			
			bias = "none",
			
			use_gradient_checkpointing = "unsloth",
			
			random_state = CFG.SEED,
		)

		return Model, Tokenizer
	
	raise ValueError("One of the inputs must be set to True")

class TrainingPrint(TrainerCallback):

	def __init__(self):
		self.StartTime = time.time()
		self.SmoothedLoss = 0

	def on_log(self, args, state, control, logs=None, **kwargs):
		VRAMused = float(torch.cuda.memory_allocated()) / 1024**3
		VRAMmax = float(torch.cuda.get_device_properties(0).total_memory) / 1024**3

		# Raw loss
		# Smoothed loss
		# Gradient norm
		# Learning Rate
		#
		# Steps / sec
		# Samples / sec
		# Tokens / sec
		# VRAM
		# 
		# Step
		# Epoch
		# Time elapsed
		# ETA
		# Progress bar
		
		

		self.SmoothedLoss = self.SmoothedLoss * 0.6   +   float(logs.get('loss', 0)) * 0.4

		FormattedString = f"""
		┌{'—'*38}┐
		|{f'RAT RNN Training Stage: {CurrentStageName}':^38}|
		├{'—'*38}┤
		|Loss (Raw):     {float(logs.get('loss', 0)):<22.3f}|
		|Loss (Smooth):  {self.SmoothedLoss:<22.3f}|
		|Gradient Norm:  {float(logs.get('grad_norm', 0)):<22.3f}|
		|Learning Rate:  {float(logs.get('learning_rate', 0)):<22.5f}|
		|{' '*38}|
		|VRAM:           {f'{VRAMused:.1f}/{VRAMmax:.1f}':<22}|
		|{' '*38}|
		|Step:           {f"{state.global_step}/{state.max_steps}":<22}|
		|Epoch:          {float(logs.get('epoch', 0)):<22.3f}|
		└{'—'*38}┘
		"""

		print(FormattedString)



def Train(Model, Tokenizer, Dataset: Dataset, StageConfig: Config.StageConfig):
	global CurrentStageName
	CurrentStageName = StageConfig.NAME

	print("Starting training...")

	OutputDir = os.path.join(CFG.CHECKPOINT_SAVE_PATH, StageConfig.NAME)
	os.makedirs(OutputDir, exist_ok=True)

	Trainer = SFTTrainer(
		model = Model,
		tokenizer = Tokenizer,
		train_dataset = Dataset,
		dataset_text_field = "text",

		max_seq_length = TCFG.MAX_SEQ_LENGTH,

		packing = True,

		args = TrainingArguments(
			output_dir = OutputDir,

			per_device_train_batch_size = TCFG.BATCH_SIZE,
			gradient_accumulation_steps = TCFG.GRAD_ACCUM,

			num_train_epochs = StageConfig.EPOCHS,
			learning_rate = StageConfig.LR,

			bf16 = TCFG.USE_BF16,
			fp16 = not TCFG.USE_BF16,

			logging_steps = 5,

			optim = "paged_adamw_8bit",
			warmup_steps = 80,

			save_strategy = "no",
			#save_steps = 20,
			# save_total_limit = 12, NO LIMIT

			report_to = "none",
		),

		#callbacks = [TrainingPrint()]
	)

	print(type(Trainer.args))
	print(type(Trainer.args).__module__)
	Trainer.train()

	# Save the final model for this stage
	OutputDir = os.path.join(CFG.MODEL_SAVE_PATH, StageConfig.NAME)
	os.makedirs(OutputDir, exist_ok=True)

	# TODO - Make the print prettier

	print("THE TRAINING FOR THIS STAGE HAS ENDED")

	Model.save_pretrained_merged(
		OutputDir,
		Tokenizer,
		save_method="merged_16bit",
	)

	print("Final model saved to:", OutputDir)

	return Model