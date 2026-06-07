from dataclasses import dataclass, field
import os


# SETTINGS
# SETTINGS
# SETTINGS
# SETTINGS
# SETTINGS
@dataclass(frozen=True)
class Config:
	"""
	Global or general configuration settings
	"""
	# Base model
	BASE_MODEL: str = "unsloth/Qwen2.5-7B-Instruct"

	# Sav paths
	MODEL_SAVE_PATH: str = os.path.expanduser("~/RATRNN/ModelSave") # ~/RATRNN/ModelSave, to enable fast loading
	CHECKPOINT_SAVE_PATH: str = "/mnt/f/Coding/AI/RATRNN_Checkpoints" # /mnt/f/Coding/AI/RATRNN_Checkpoints, so it doesn't flood my L (Linux) partition, it's only 100 GB
	TRAINING_DATA_PATH: str = os.path.expanduser("~/RATRNN/TrainingData") # ~/RATRNN/TrainingData
	DATASET_CACHE_PATH: str = os.path.expanduser("~/RATRNN/TrainingData/Cache") # ~/RATRNN/TrainingData

	LOAD_IN_4BIT: bool = True # QLoRA bit mode (4 / 8)
	LOAD_IN_8BIT: bool = False

	# System
	SEED: int = 52

@dataclass(frozen=True)
class TrainingConfig:
	"""
	Settings purely used in the trainer
	"""
	MAX_SEQ_LENGTH: int = 1024
	LABEL_SMOOTHING: float = 0.03

	# LoRA config
	LORA_R: int = 32
	LORA_ALPHA: int = 64
	LORA_DROPOUT: float = 0.05

	LORA_TARGET_MODULES: list[str] = field(default_factory=lambda: [
		"q_proj", "k_proj", "v_proj", "o_proj",
		"gate_proj", "up_proj", "down_proj"
	])

	# Precision
	USE_BF16: bool = True

	# Training
	BATCH_SIZE: int = 1
	GRAD_ACCUM: int = 12
	LR: float = 2e-4
	EPOCHS: int = 1

	# Multi-stage training
	TRAINING_STAGES: list[str] = field(default_factory=lambda: [""
		"Stage1"
	])

@dataclass(frozen=True)
class InferenceConfig:
	MAX_SEQ_LENGTH: int = 4096
	MAX_OUTPUT_TOKENS: int = 256



# STAGE PRESET
# STAGE PRESET
# STAGE PRESET
# STAGE PRESET
# STAGE PRESET
@dataclass(frozen=True)
class StageConfig:
	NAME: str = "Stage"

	MESSAGE_HISTORY_LENGTH: int = 10
	MAX_LENGTH_BIAS: float = 0.2 # 0-1, controls the percentage of data that is at a certain message history length. E.g.
	# - MAX_LENGTH_BIAS = 1, so only message histories with a lengt of MESSAGE_HISTORY_LENGTH are used
	# - MAX_LENGTH_BIAS = 0.5, the chance is different for each value, is a blend between 1 and 0, using a math formula
	# - MAX_LENGTH_BIAS = 0, so all message histories have an equal chance to be used (1/MESSAGE_HISTORY_LENGTH to preserve scale)

	DATASET_SCALE: float = 1 # Works best when 0-1, simply multiplies the chance of something to make it in the dataset by this value.
	# Meant to work as a shrinking factor, but can poorly work as a scaling facor if you set fractional dataset contents below.

	# Absolute is there whe contents are just percentages of each dataset
	# Relative is where the contents are based off of the smallest size (sum to 100%)
	# Both get added together afterwards WARNING: potential duplicated data
	ABSOLUTE_DATASET_CONTENTS: dict = field(default_factory=lambda: {
		"AllUserMessages": 0,
		"MyMessages": 0,
		"FineTuning": 1
	})
	
	RELATIVE_DATASET_CONTENTS: dict = field(default_factory=lambda: {
		"AllUserMessages": 0.8,
		"MyMessages": 0.2,
		"FineTuning": 0
	})

	TRAINING_MODE: str = "FullConversation" # "FullConversation" | "AssistantOnly" | "MostRecent"
	# FullConversation: Trains EVERY message token, except for messages sent by the "system" and "tool" roles,   role learning issues but fastest information gathering
	# AssistantOnly: Trains ONLY on assistant messages,   potential overfit risk
	# MostRecent: Trains ONLY on the final assistant message, most stable,   slowest learning

	BASE_TOKEN_WEIGHTS: dict = field(default_factory=lambda: { # By role
		"system": 0,
		"tool": 0,
		"user": 0.4,
		"assistant": 1
	})
	TOKEN_DECAY_RATE: float = 0 # 0-1, 0 for all tokens to keep their original weight, 1 will decay token values based off of recency
	



















# TRAINING STAGES
# TRAINING STAGES
# TRAINING STAGES
# TRAINING STAGES
# TRAINING STAGES
Stage1 = StageConfig(
	NAME = "Stage 1",

	MESSAGE_HISTORY_LENGTH = 10,
	MAX_LENGTH_BIAS = 0.2,

	DATASET_SCALE = 1,

	ABSOLUTE_DATASET_CONTENTS = {
		"AllUserMessages": 0,
		"MyMessages": 0,
		"FineTuning": 1
	},
	RELATIVE_DATASET_CONTENTS = {
		"AllUserMessages": 0.8,
		"MyMessages": 0.2,
		"FineTuning": 0
	},

	TRAINING_MODE = "FullConversation",
	BASE_TOKEN_WEIGHTS = {"system": 0, "tool": 0, # By role
		"user": 0.4,
		"assistant": 1
	},
	TOKEN_DECAY_RATE = 0
)

Stage2 = StageConfig(
	MESSAGE_HISTORY_LENGTH = 15,
	MAX_LENGTH_BIAS = 0.7,

	DATASET_SCALE = 1,

	ABSOLUTE_DATASET_CONTENTS = {
		"AllUserMessages": 0,
		"MyMessages": 0,
		"FineTuning": 1
	},
	RELATIVE_DATASET_CONTENTS = {
		"AllUserMessages": 0.1,
		"MyMessages": 1,
		"FineTuning": 0
	},

	TRAINING_MODE = "FullConversation",
	BASE_TOKEN_WEIGHTS = {"system": 0, "tool": 0, # By role
		"user": 0.4,
		"assistant": 1
	},
	TOKEN_DECAY_RATE = 0
)