import torch
import lightning as L
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.nn import functional as F
from constants import TARGET_MODEL, DRAFT_MODEL
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from datamodule import WikiTextV2Datamodule
import os
from constants import TARGET_MODEL, DRAFT_MODEL
import torch
import lightning as L


class DraftModelFinetuner(L.LightningModule):
    def __init__(
        self,
        draft_model_name=DRAFT_MODEL,
        target_model_name=TARGET_MODEL,
        learning_rate=5e-5,
        weight_decay=0.01,
    ):
        super().__init__()
        
        self.save_hyperparameters()
    
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name, 
            torch_dtype=torch.float16
        )
        
        # TODO Мб только токенайзер от нее нужен, т.к. предпосчет мы делаем в другом месте
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name, 
            torch_dtype=torch.float16
        )
        
        for param in self.target_model.parameters():
            param.requires_grad = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        target_scores = batch["scores"]
        
        draft_outputs = self.draft_model(input_ids)
        draft_logits = draft_outputs.logits[:, -1, :]
        
        log_draft_probs = F.log_softmax(draft_logits, dim=-1)
        target_probs = F.softmax(target_scores, dim=-1)    
        
        loss = F.kl_div(log_draft_probs, target_probs, reduction='batchmean')
        self.log("loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
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


if __name__ == "__main__":
   
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = "cuda:4"

    #target_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device)
    #draft_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.float16).to(device)
    draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL, torch_dtype=torch.float16).to(device)
    target_model.eval()
    draft_model.train()
    
    datamodule = WikiTextV2Datamodule(
        min_len=5,  
        max_len=100,
        target_model=target_model,
        device=device,
        batch_size=1, 
    )
    datamodule.setup(stage="fit")
    torch.set_float32_matmul_precision('medium')

    trainer = L.Trainer(
        accelerator="gpu", max_epochs=1, 
        precision=16,
        limit_train_batches=None,
        logger=False, # TensorBoardLogger(save_dir=".")
        devices=[4] 
    )

    finetuner = DraftModelFinetuner()
    trainer.fit(model=finetuner, datamodule=datamodule)