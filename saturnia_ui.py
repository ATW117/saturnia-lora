#!/usr/bin/env python3
"""Local FLUX.2 Klein UI for Saturnia LoRAs."""
from __future__ import annotations
import argparse, gc, json, os, random, sys, threading, time
from dataclasses import dataclass
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
ROOT = Path(__file__).resolve().parent
DEFAULT_TOOLKIT = ROOT.parent / "ai-toolkit"
OUTPUT_DIR = ROOT / ".runs" / "ui_generations"
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT / '.runs' / 'compiler_cache'))
NO_LORA = "None (base model)"
@dataclass(frozen=True)
class LoraChoice:
    label: str
    path: Path
def discover_loras(root: Path = ROOT) -> list[LoraChoice]:
    out=[]
    for path in sorted((root/".runs"/"output").glob("*/*.safetensors")):
        run=path.parent.name; suffix=path.stem.removeprefix(run).lstrip("_")
        checkpoint=f"step {int(suffix):,}" if suffix.isdigit() else "final"
        out.append(LoraChoice(f"{run} — {checkpoint}", path.resolve()))
    return out
def _safe_dimensions(w,h):
    w,h=int(w),int(h)
    if not (256<=w<=1536 and 256<=h<=1536): raise ValueError("Dimensions must be 256–1536.")
    return w//16*16,h//16*16

def browser_image(image):
    """Smaller transport file with exactly the same pixels; retain original PNG."""
    from PIL import Image
    preview = image.with_name('display.webp')
    with Image.open(image) as source:
        source.save(preview, format='WEBP', lossless=True, method=4)
    return preview if preview.stat().st_size < image.stat().st_size else image
class KleinEngine:
    def __init__(self, toolkit, compile_lora=None):
        self.toolkit=Path(toolkit).resolve(); self.model=self.network=self.pipeline=None
        if str(self.toolkit) not in sys.path: sys.path.insert(0, str(self.toolkit))
        self.loaded_lora=None; self.lock=threading.Lock()
        self.optimized = True
        from collections import OrderedDict
        self.prompt_cache = OrderedDict()
        self.output_dir = OUTPUT_DIR
        self.last_seconds = None
        self.compile_lora = (os.environ.get('SATURNIA_COMPILE_LORA', '1') == '1'
                             if compile_lora is None else compile_lora)
        self.compiled_lora = None
    def load(self):
        if self.model is not None: return
        if not (self.toolkit/"toolkit").is_dir(): raise RuntimeError(f"AI Toolkit not found: {self.toolkit}")
        sys.path.insert(0,str(self.toolkit))
        import torch
        from extensions_built_in.diffusion_models.flux2.flux2_klein_model import Flux2Klein4BModel
        from toolkit.config_modules import ModelConfig
        from toolkit.lora_special import LoRASpecialNetwork
        cfg=ModelConfig(name_or_path="black-forest-labs/FLUX.2-klein-base-4B",arch="flux2_klein_4b",dtype="bf16",quantize=False,quantize_te=False,low_vram=False)
        model=Flux2Klein4BModel(device="cuda:0",model_config=cfg,dtype="bf16"); model.load_model()
        net=LoRASpecialNetwork(text_encoder=model.text_encoder,unet=model.get_model_to_train(),lora_dim=32,multiplier=1.,alpha=32,train_unet=True,train_text_encoder=False,conv_lora_dim=None,conv_alpha=None,is_sdxl=False,is_v2=False,is_v3=False,is_pixart=False,is_auraflow=False,is_flux=False,is_lumina2=False,is_ssd=False,is_vega=False,dropout=None,use_text_encoder_1=True,use_text_encoder_2=True,use_bias=False,is_lorm=False,network_config=None,network_type="lora",transformer_only=False,is_transformer=True,base_model=model,target_lin_modules=model.target_lora_modules)
        net.force_to(model.device_torch,dtype=torch.float32); net.apply_to(model.text_encoder,model.get_model_to_train(),False,True); net.can_merge_in=False
        model.network=net; self.model,self.network=model,net; self.pipeline=model.get_generation_pipeline()
        if self.compile_lora:
            from compiled_lora import CompiledLora
            self.compiled_lora = CompiledLora(net)
            self.compiled_lora.enabled = True
    def select(self,path,strength):
        import torch
        if path is None: self.network.multiplier=0.
        else:
            if path!=self.loaded_lora:
                self.network.load_weights(str(path)); self.network.force_to(self.model.device_torch,dtype=torch.float32); self.loaded_lora=path
            self.network.multiplier=float(strength)
        self.network._update_torch_multiplier()
    def generate(self,prompt,path,strength,seed,w,h,steps,guidance):
        import torch
        from toolkit.config_modules import GenerateImageConfig
        prompt=prompt.strip()
        if not prompt: raise ValueError("Enter a prompt.")
        w,h=_safe_dimensions(w,h); steps=int(steps); seed=random.SystemRandom().randrange(2**31) if int(seed)<0 else int(seed)
        if not 1<=steps<=100: raise ValueError("Steps must be 1–100.")
        with self.lock,torch.no_grad():
            self.load(); self.select(path,strength); self.output_dir.mkdir(parents=True,exist_ok=True)
            started = time.perf_counter()
            work=self.output_dir/f"{time.strftime('%Y%m%d-%H%M%S')}-seed{seed}"; n=1
            while work.exists(): work=self.output_dir/f"{time.strftime('%Y%m%d-%H%M%S')}-seed{seed}-{n}"; n+=1
            work.mkdir()
            cfg=GenerateImageConfig(prompt=prompt,width=w,height=h,num_inference_steps=steps,guidance_scale=float(guidance),negative_prompt="",seed=seed,network_multiplier=(float(strength) if path else 0.0),output_ext="png",output_folder=str(work))
            if self.optimized:
                self.generate_direct(cfg)
            else:
                self.model.generate_images([cfg],pipeline=self.pipeline)
            images=sorted(work.glob("*.png"))+sorted(work.glob("*.jpg"))
            if not images: raise RuntimeError(f"No image found in {work}")
            image=images[-1]; meta=dict(prompt=prompt,seed=seed,width=w,height=h,steps=steps,guidance=float(guidance),lora=str(path) if path else None,lora_strength=float(strength) if path else 0,image=image.name)
            self.last_seconds = time.perf_counter() - started
            meta['generation_seconds'] = self.last_seconds
            meta['compiled_lora'] = bool(self.compiled_lora and self.compiled_lora.enabled)
            (work/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
            return image,seed

    def generate_direct(self, cfg):
        """Use the original sampler without training-state cleanup or repeated encoding."""
        import torch
        import copy
        self.network.eval()
        self.model.model.eval()
        self.model.vae.eval()
        for encoder in self.model.text_encoder:
            encoder.eval()
        with self.network:
            self.network.multiplier = cfg.network_multiplier
            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed(cfg.seed)
            generator = torch.manual_seed(cfg.seed)
            self.model.prepare_sample_prompt_context(cfg)
            embeds = []
            for text in (cfg.prompt, cfg.negative_prompt):
                if text not in self.prompt_cache:
                    self.prompt_cache[text] = self.model.encode_prompt(
                        text, None, force_all=True, control_images=None)
                    while len(self.prompt_cache) > 16:
                        self.prompt_cache.popitem(last=False)
                self.prompt_cache.move_to_end(text)
                embeds.append(copy.deepcopy(self.prompt_cache[text]).to(
                    self.model.device_torch, dtype=self.model.unet.dtype))
            cfg.post_process_embeddings(*embeds)
            img = self.model.generate_single_image(
                self.pipeline, cfg, *embeds, generator, {})
            cfg.save_image_atomic(img, 0)
def build_app(toolkit):
    import gradio as gr
    choices=discover_loras(); paths={x.label:x.path for x in choices}; engine=KleinEngine(toolkit)
    def run(prompt,label,strength,seed,w,h,steps,guidance):
        image,used=engine.generate(prompt,paths.get(label),strength,seed,w,h,steps,guidance)
        return str(browser_image(image)),used,f"Generated in {engine.last_seconds:.1f}s. Saved to {image.parent}",str(image)
    with gr.Blocks(title="Saturnia — FLUX.2 Klein") as app:
        gr.Markdown("# Saturnia — FLUX.2 Klein\nLocal inference; downloads are disabled.")
        with gr.Row():
            with gr.Column(scale=2):
                prompt=gr.Textbox(label="Prompt",lines=5,value="SATURNIA_STYLE. A small botanical moth creature on warm paper.")
                lora=gr.Dropdown([NO_LORA]+[x.label for x in choices],value=NO_LORA,label="LoRA checkpoint")
                strength=gr.Slider(0,1.5,.8,step=.05,label="LoRA strength")
                with gr.Row(): seed=gr.Number(-1,precision=0,label="Seed (-1 = random)"); steps=gr.Slider(1,60,28,step=1,label="Steps"); guidance=gr.Slider(1,8,4,step=.1,label="Guidance")
                with gr.Row(): w=gr.Dropdown([512,768,1024,1280],value=1024,label="Width"); h=gr.Dropdown([512,768,1024,1280],value=1024,label="Height")
                button=gr.Button("Generate",variant="primary")
            with gr.Column(scale=3): output=gr.Image(label="Generated image",type="filepath"); used=gr.Number(label="Used seed",precision=0); status=gr.Textbox(label="Status",interactive=False)
            original = gr.File(label="Original PNG", interactive=False)
        button.click(run,[prompt,lora,strength,seed,w,h,steps,guidance],[output,used,status,original])
    return app
def main():
    p=argparse.ArgumentParser(); p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=7860); p.add_argument("--toolkit-dir",type=Path,default=DEFAULT_TOOLKIT); p.add_argument("--list-loras",action="store_true"); a=p.parse_args()
    if a.list_loras:
        for x in discover_loras(): print(f"{x.label}\t{x.path}")
    else: build_app(a.toolkit_dir).queue(default_concurrency_limit=1).launch(server_name=a.host,server_port=a.port,share=False,show_error=True)
if __name__=="__main__": main()
