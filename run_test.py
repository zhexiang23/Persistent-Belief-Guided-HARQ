"""Kodak joint-selection and Figure 11 erasure inference. Compiled model implementations."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ('runtime_py39' if sys.version_info[:2]==(3,9) else 'runtime')))
os.environ['TORCH_HOME']=str(ROOT/'metric_weights')
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from core_tokcom.data import encode_bytes
from evaluate_impl import load_protocol_image,tiles_from_image,stitch
from joint_harq_evaluation import metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(kind,channel,device):
    if kind=='r025':
        from adaptive_rate_model import AdaptiveRateHARQ,AdaptiveRateConfig
        path=ROOT/'weights'/f'R025_{channel}_epoch100_ema.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=AdaptiveRateHARQ(AdaptiveRateConfig(**state['model_config']))
        model.load_state_dict(state['model'],strict=True);model.set_rate(.25)
    else:
        if device!='cuda':raise ValueError('figure11 currently requires CUDA')
        import erasure_runtime as e
        path=ROOT/'weights/v30_epoch22.pt';e.CKPT=path
        model=e.read_model('legacy',16,1);model.mode='CoRe-TokCom'
    return model.to(device).eval().requires_grad_(False),path


@torch.inference_mode()
def infer(model,source,kind,channel,snr,rounds,batch_size,seed,device,erased=0):
    if kind=='figure11':
        import erasure_runtime as e
        model.erased=erased
        text=encode_bytes('a high resolution natural image',model.cfg.text_length).unsqueeze(0)
        return e.reconstruct(model,source,text,seed,rounds)
    patches,nr,nc=tiles_from_image(source,model.cfg.image_size)
    parts=[[] for _ in range(rounds+1)];uses=[0.]*(rounds+1)
    text=encode_bytes('a high resolution natural image',model.cfg.text_length).unsqueeze(0).to(device)
    for offset in range(0,len(patches),batch_size):
        torch.manual_seed(seed+offset)
        batch=patches[offset:offset+batch_size].to(device)
        s=torch.full((len(batch),),float(snr),device=device)
        out=model.forward_harq(batch,text.expand(len(batch),-1),s,channel,rounds)
        imgs=out['images'];costs=out['uses']
        for t in range(rounds+1):parts[t].append(imgs[t].cpu());uses[t]+=float(costs[t].sum())
    return [stitch(torch.cat(v),nr,nc) for v in parts],uses


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',choices=['r025','token-query','figure11'])
    p.add_argument('--dataset',choices=['CelebA-HQ','Kodak'],default=None)
    for key in ['celeba-hq','kodak']:p.add_argument('--'+key+'-root',type=Path)
    p.add_argument('--manifest',type=Path,default=ROOT/'configs/figure11_images.json')
    p.add_argument('--channels',nargs='+',choices=['awgn','rayleigh'],default=None)
    p.add_argument('--snr-db',nargs='+',type=float,default=None)
    p.add_argument('--retransmissions',type=int,default=None)
    p.add_argument('--image-limit',type=int)
    p.add_argument('--batch-size',type=int,default=None)
    p.add_argument('--trials',type=int,default=None)
    p.add_argument('--erased',nargs='+',type=int,default=[0,4,8,12,16])
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.experiment=='token-query':a.experiment='figure11'
    if a.trials is None:a.trials=5 if a.experiment=='figure11' else 1
    if a.experiment=='figure11' and sys.version_info[:2]!=(3,9):p.error('Original erasure protocol requires Python 3.9 + PyTorch 1.13; see requirements_erasure.txt')
    if a.experiment=='r025' and sys.version_info[:2]!=(3,12):p.error('R025 requires Python 3.12 + PyTorch 2.5.1')
    if a.experiment=='figure11' and not torch.__version__.startswith('1.13.'):p.error('figure11 requires PyTorch 1.13.x')
    if a.experiment=='r025' and not torch.__version__.startswith('2.5.1'):p.error('r025 requires PyTorch 2.5.1')
    if a.experiment=='figure11' and a.device!='cuda':p.error('figure11 requires CUDA')
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA is unavailable in this environment')
    if a.image_limit is not None and a.image_limit<1:p.error('image-limit must be positive')
    if a.trials<1:p.error('trials must be positive')
    if a.experiment=='r025':
        channels=a.channels or ['awgn','rayleigh'];snrs=a.snr_db or list(range(-6,19,2));rounds=3 if a.retransmissions is None else a.retransmissions
        batch=a.batch_size or 32;datasets=['Kodak']
        if a.dataset not in [None,'Kodak']:p.error('R=0.25 uses Kodak')
        if not 0<=rounds<=3:p.error('r025 supports 0..3 retransmissions')
    else:
        channels=a.channels or ['rayleigh'];snrs=a.snr_db or [3.5];rounds=4 if a.retransmissions is None else a.retransmissions
        batch=a.batch_size or 2
        if a.dataset not in [None,'CelebA-HQ']:p.error('Figure 11 reference protocol uses CelebA-HQ')
        datasets=['CelebA-HQ']
        if channels!=['rayleigh'] or snrs!=[3.5]:p.error('Figure 11 protocol: Rayleigh, 3.5 dB')
        if rounds!=4:p.error('figure11 reference protocol requires 4 retransmissions')
        if a.experiment=='figure11' and batch!=2:p.error('figure11 reference patch batch size is 2')
    if batch<1 or any(not 0<=n<=32 for n in a.erased):p.error('invalid batch/erasure count')
    if not all(__import__('math').isfinite(s) for s in snrs):p.error('SNR must be finite')
    roots={'CelebA-HQ':a.celeba_hq_root,'Kodak':a.kodak_root}
    files={}
    for dataset in datasets:
        root=roots[dataset]
        if root is None:p.error('missing data root for '+dataset)
        if dataset=='Kodak':paths=sorted(root.glob('kodim*.png'))
        else:
            manifest=json.loads(a.manifest.read_text())
            names=manifest['images']
            if manifest['dataset']!=dataset or len(names)!=len(set(names)):raise ValueError('invalid image manifest')
            paths=[root/Path(n.replace('\\','/')) for n in names]
        paths=paths[:a.image_limit]
        if not paths or any(not f.is_file() for f in paths):raise FileNotFoundError('missing images for '+dataset)
        files[dataset]=paths
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    protocol={**vars(a),'channels':channels,'snr_db':snrs,'rounds':rounds,'batch_size':batch,'selection':('original V30 receiver score' if a.experiment=='figure11' else 'source utility + channel reliability joint selection'),'full_path':'image encoder -> noisy channel -> HARQ combining -> image decoder -> RGB metrics','lpips':'AlexNet v0.1, normalized [-1,1] inputs','initial_cbr':.25 if a.experiment=='r025' else .004201253255208333,'weights':{},'input_sha256':{d:{str(f.relative_to(roots[d])):sha(f) for f in fs} for d,fs in files.items()}}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2,default=str),encoding='utf-8')
    import lpips
    perceptual=lpips.LPIPS(net='alex').to(a.device).eval()
    fields=['dataset','image','channel','snr_db','trial','erased','t','psnr_db','ms_ssim','lpips','forward_symbols','normalized_cost','cbr','cpp']
    sums={}
    with (a.output/'per_image_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for channel in channels:
            model,weight=load_model(a.experiment,channel,a.device);protocol['weights'][channel]=sha(weight)
            for snr in snrs:
                for dataset,paths in files.items():
                    for ix,path in enumerate(paths):
                        source=(pil_to_tensor(Image.open(path).convert('RGB')).float()/255 if dataset=='Kodak' else load_protocol_image(path,dataset))
                        for trial in range(a.trials):
                            for erased in (a.erased if a.experiment=='figure11' else [0]):
                                seed=(20260723 if a.experiment=='r025' else 20260815)+ix*100000+trial*1000+(1_000_000_000 if a.experiment=='figure11' else 0)
                                pics,uses=infer(model,source,a.experiment,channel,snr,rounds,batch,seed,a.device,erased)
                                for t,pic in enumerate(pics):
                                    row=dict(dataset=dataset,image=str(path.relative_to(roots[dataset])),channel=channel,snr_db=snr,trial=trial,erased=erased,t=t,**metrics(source,pic,perceptual,a.device),forward_symbols=float(uses[t]),normalized_cost=float(uses[t]/uses[0]),cbr=float(uses[t]/source.numel()),cpp=float(uses[t]/(source.shape[-2]*source.shape[-1])))
                                    writer.writerow(row)
                                    key=(dataset,channel,snr,erased,t);total=sums.setdefault(key,{'observations':0,**dict.fromkeys(fields[7:],0.)});total['observations']+=1
                                    for k in fields[7:]:total[k]+=row[k]
                                f.flush();print(dataset,channel,snr,path.name,'trial',trial,'erased',erased,'completed',flush=True)
            del model
            if torch.cuda.is_available():torch.cuda.empty_cache()
    with (a.output/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','channel','snr_db','erased','t','observations',*fields[7:]]);writer.writeheader()
        for key,total in sums.items():writer.writerow(dict(zip(['dataset','channel','snr_db','erased','t'],key))|{'observations':total['observations']}|{k:total[k]/total['observations'] for k in fields[7:]})
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2,default=str),encoding='utf-8')
    (a.output/'complete.json').write_text(json.dumps({'completed':True,'rows':sum(v['observations'] for v in sums.values()),'csv_sha256':sha(a.output/'per_image_metrics.csv')},indent=2))

if __name__=='__main__':main()
