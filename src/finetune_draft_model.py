import torch
import lightning as L
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
from constants import CUDA_DEVICE, DRAFT_MODEL, EPS, TARGET_MODEL, SMALL_TARGET_MODEL, SMALL_DRAFT_MODEL
from transformers import AutoModelForCausalLM, AutoTokenizer
from datamodule import WikiTextV2Datamodule
import os
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import lightning as L
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class Lit(L.LightningModule):
    def __init__(
        self, draft_model, learning_rate=1e-6, weight_decay=0.01, temperature=2.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['draft_model'])
        self.draft_model = draft_model
        self.draft_model.train()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature 
    
    def forward(self, *args, **kwargs):
        return self.draft_model(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        target_logits = batch["logits"]
        
        # Trying to normalize target logits
        scaled_target_logits = target_logits / self.temperature
        
        draft_logits = self.draft_model(input_ids).logits
        log_draft_probs = F.log_softmax(draft_logits, dim=-1)
        target_probs = F.softmax(scaled_target_logits, dim=-1)    
        
        loss = F.kl_div(log_draft_probs, target_probs, reduction='batchmean')
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        target_logits = batch["logits"]
    
        scaled_target_logits = target_logits / self.temperature
        
        draft_logits = self.draft_model(input_ids).logits
        log_draft_probs = F.log_softmax(draft_logits, dim=-1).clamp(min=EPS)
        target_probs = F.softmax(scaled_target_logits, dim=-1).clamp(min=EPS)
        
        loss = F.kl_div(log_draft_probs, target_probs, reduction='batchmean')
        self.log("val_loss", loss, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.draft_model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

def create_peft_config(model):
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return get_peft_model(prepare_model_for_kbit_training(model), peft_config)

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('medium')
    L.seed_everything(42)

    target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL).to(CUDA_DEVICE)
    target_model_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    draft_model = create_peft_config(AutoModelForCausalLM.from_pretrained(DRAFT_MODEL).to(CUDA_DEVICE))
    
    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=False),
    ]
    
    datamodule = WikiTextV2Datamodule(
        min_len=50,  
        max_len=175,
        target_model=target_model,
        target_model_tokenizer=target_model_tokenizer,
        device=CUDA_DEVICE,
        batch_size=2,
        num_workers=25
    )

    trainer = L.Trainer(
        accelerator="gpu", max_epochs=15, 
        limit_train_batches=None,
        logger=False,
        devices=[int(CUDA_DEVICE.split(":")[-1])],
        callbacks=callbacks,
    )

    fine_tuned_model = Lit(draft_model=draft_model, learning_rate=1e-6)
    trainer.fit(model=fine_tuned_model, datamodule=datamodule)
    
    torch.save(fine_tuned_model.state_dict(), "fine_tuned_weights.pth")