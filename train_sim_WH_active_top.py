from pathlib import Path
import time
import torch
import numpy as np
import math
from functools import partial
from itertools import islice
from dataset import WHDataset, seed_worker
from torch.utils.data import DataLoader
from transformer_sim import Config, TSTransformer
from train_utils import warmup_cosine_lr
import tqdm
import argparse
import wandb
import copy
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Meta system identification with transformers')

    # Overall
    parser.add_argument('--model-dir', type=str, default="check_patch_WH/active_430_400skip_decoder", metavar='S',
                        help='Saved model folder')
    parser.add_argument('--out-file', type=str, default="resume_top32_over80_fixed_err0.01_800k_seed2", metavar='S',
                        help='Saved model name')
    parser.add_argument('--in-file', type=str, default="ckpt_top32_err0.01_100k", metavar='S',
                        help='Loaded model name (when resuming)')
    parser.add_argument('--init-from', type=str, default="resume", metavar='S',
                        help='Init from (scratch|resume|pretrained)')
    parser.add_argument('--seed', type=int, default=43, metavar='N',
                        help='Seed for random number generation')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='Use wandb for data logging')

    # Dataset
    parser.add_argument('--nx', type=int, default=5, metavar='N',
                        help='Model order nx (default: 5)')
    parser.add_argument('--nu', type=int, default=1, metavar='N',
                        help='Number of inputs nu (default: 1)')
    parser.add_argument('--ny', type=int, default=1, metavar='N',
                        help='Number of outputs ny (default: 1)')
    parser.add_argument('--seq-len-skip', type=int, default=400, metavar='N',
                        help='sequence length of the separation between context and query (default: 0)')
    parser.add_argument('--seq-len-n-in', type=int, default=30, metavar='N',
                        help='sequence length of the initial condition previous to the query (default: 0)')
    parser.add_argument('--seq-len-ctx', type=int, default=400, metavar='N',
                        help='sequence length of the context sequence (default: 400)')
    parser.add_argument('--seq-len-new', type=int, default=100, metavar='N',
                        help='sequence length of the query one (default: 100)')
    parser.add_argument('--seq-len-patch', type=int, default=400, metavar='N',
                    help='number of patches (default: 300)')
    parser.add_argument('--mag_range', type=tuple, default=(0.5, 0.97), metavar='N',
                        help='range of poles magnitude (default: (0.5, 0.97))')
    parser.add_argument('--phase_range', type=tuple, default=(0.0, math.pi/2), metavar='N',
                        help='range of poles phase (default: (0.0, math.pi/2))')
    parser.add_argument('--fixed-system', action='store_true', default=False,
                        help='If True, keep the same model all the times')

    # Model
    parser.add_argument('--n-layer', type=int, default=12, metavar='N',
                        help='Number of layers (default: 12)')
    parser.add_argument('--n-head', type=int, default=4, metavar='N',
                        help='Number heads (default: 4)')
    parser.add_argument('--d-model-RNN', type=int, default=128, metavar='N',
                        help='number of embedding from the patching (default: 1M)')
    parser.add_argument('--n-embd', type=int, default=128, metavar='N',
                        help='Embedding size (default: 128)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='LR',
                        help='Dropout (default: 0.0)')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Use bias in model')

    # Training
    parser.add_argument('--batch-size', type=int, default=80, metavar='N',
                        help='Batch size (default:32)')
    parser.add_argument('--max-iters', type=int, default=700_100, metavar='N',
                        help='Number of iterations (default: 1000000)')
    parser.add_argument('--warmup-iters', type=int, default=10_000, metavar='N',
                        help='Number of warmup iterations (default: 10000)')
    parser.add_argument('--lr', type=float, default=6e-4, metavar='LR',
                        help='Learning rate (default: 6e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='D',
                        help='Optimizer weight decay (default: 0.0)')
    parser.add_argument('--eval-interval', type=int, default=2000, metavar='N',
                        help='Frequency of performance evaluation (default:2000)')
    parser.add_argument('--eval-iters', type=int, default=100, metavar='N',
                        help='Number of batches used for performance evaluation')
    parser.add_argument('--fixed-lr', action='store_true', default=False,
                        help='Keep the learning rate constant, do not use cosine scheduling')

    # Compute
    parser.add_argument('--threads', type=int, default=10,
                        help='Number of CPU threads (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--cuda-device', type=str, default="cuda:3", metavar='S',
                        help='Cuda device (default: "cuda:2")')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Compile the model with torch.compile')
    try:
        cfg = parser.parse_args()

        # Other settings
        cfg.beta1 = 0.9
        cfg.beta2 = 0.95

        # Derived settings
        #cfg.block_size = cfg.seq_len
        cfg.lr_decay_iters = cfg.max_iters
        cfg.min_lr = cfg.lr/10.0  #
        cfg.decay_lr = not cfg.fixed_lr
        cfg.eval_batch_size = cfg.batch_size

        # Init wandb
        
        if cfg.log_wandb:
            wandb.init(
                project="sysid-meta-WH-active",
                name="resume_top32_over80_fixed_err01_seed2",
                # track hyperparameters and run metadata
                config=vars(cfg)
            )

        # Set seed for reproducibility
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed) # not needed? All randomness now handled with generators

        # Create out dir
        model_dir = Path(cfg.model_dir)
        model_dir.mkdir(exist_ok=True)

        # Configure compute
        # import multiprocessing
        # multiprocessing.set_start_method('spawn', force=True)

        torch.set_num_threads(cfg.threads)
        use_cuda = not cfg.no_cuda and torch.cuda.is_available()
        device_name = cfg.cuda_device if use_cuda else "cpu"
        device = torch.device(device_name)
        device_type = 'cuda' if 'cuda' in device_name else 'cpu'
        torch.set_float32_matmul_precision("high")

        lin_opts = dict(mag_range=cfg.mag_range, phase_range=cfg.phase_range, strictly_proper=True)
        mdlargs = dict(mag_range=cfg.mag_range, phase_range=cfg.phase_range)#, strictly_proper=True)
        train_ds = WHDataset(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, 
                                            seq_len=cfg.seq_len_ctx+cfg.seq_len_skip+cfg.seq_len_n_in+cfg.seq_len_new,
                            system_seed=cfg.seed, input_seed=cfg.seed+1, noise_seed=cfg.seed+2,  
                            **lin_opts)
        # train_ds = BenchmarkDataset(seq_len_ctx = cfg.seq_len_ctx, seq_len_new = cfg.seq_len_new, seq_len_n_in = cfg.seq_len_n_in,system_seed=cfg.seed+2, input_seed=cfg.seed+3, noise_seed=cfg.seed+4)

        # train_dl = wh_dataset(seq_len=cfg.seq_len_ctx+cfg.seq_len_skip+cfg.seq_len_n_in+cfg.seq_len_new, batch_size=cfg.batch_size, nx=cfg.nx, **mdlargs)
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.threads, worker_init_fn=seed_worker)

        # if we work with a constant model we also validate with the same (thus same seed!)
        val_ds = WHDataset(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, 
                                            seq_len=cfg.seq_len_ctx+cfg.seq_len_skip+cfg.seq_len_n_in+cfg.seq_len_new,
                        system_seed=cfg.seed+5, input_seed=cfg.seed+6, noise_seed=cfg.seed+7, 
                        **lin_opts)
        # val_ds = BenchmarkDataset(seq_len_ctx = cfg.seq_len_ctx, seq_len_new = cfg.seq_len_new, seq_len_n_in = cfg.seq_len_n_in,system_seed=cfg.seed+5, input_seed=cfg.seed+6, noise_seed=cfg.seed+7)

        val_dl = DataLoader(val_ds, batch_size=cfg.eval_batch_size, num_workers=cfg.threads,worker_init_fn=seed_worker)

        model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, n_y=cfg.ny, n_u=cfg.nu,
                            seq_len_ctx=cfg.seq_len_ctx, seq_len_new=cfg.seq_len_new,d_model_RNN = cfg.d_model_RNN,
                            bias=cfg.bias, dropout=cfg.dropout, seq_len_patch = cfg.seq_len_patch, device_name = cfg.cuda_device)
        
        if cfg.init_from == "scratch":
            gptconf = Config(**model_args)
            model = TSTransformer(gptconf)
            best_model = copy.deepcopy(model).to(device)
            # model.load_state_dict(state_dict)

        elif cfg.init_from == "resume" or cfg.init_from == "pretrained":
            ckpt_path = model_dir / f"{cfg.in_file}.pt"
            checkpoint = torch.load(ckpt_path, map_location=device)
            gptconf = Config(**checkpoint["model_args"])
            model = TSTransformer(gptconf)
            state_dict = checkpoint['model']
            model.load_state_dict(state_dict)
            best_model = copy.deepcopy(model).to(device)
            

            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            #unwanted_prefix = '_orig_mod.'
            #for k, v in list(state_dict.items()):
            #    if k.startswith(unwanted_prefix):
            #        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.to(device)

        if cfg.compile:
            model = torch.compile(model)  # requires PyTorch 2.0


        optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)

        if cfg.init_from == "resume":
            optimizer.load_state_dict(checkpoint['optimizer'])

        vec_choice = [40,35,30,25]
        iteration = [700000]#00,5000,25000]
        max_iter = sum(iteration)
        @torch.no_grad()
        def estimate_loss(i):
            model.eval()
            loss = 0.0
            rmse = 0.0
            loss_2 = 0.0
            rmse_2 = 0.0
            #Doing this you compute the loss all over the validation set 
            for eval_iter, (batch_y, batch_u) in enumerate(val_dl):
                # print(batc_y.shape)
                batch_y_enc_fake = batch_y[:,:cfg.seq_len_ctx,:]
                batch_u_enc_fake = batch_u[:,:cfg.seq_len_ctx,:]
                batch_y_dec_fake = batch_y[:,cfg.seq_len_ctx+cfg.seq_len_skip:,:]
                batch_u_dec_fake = batch_u[:,cfg.seq_len_ctx+cfg.seq_len_skip:,:]
                # batch_y_dec = batch_y[:,cfg.seq_len_ctx+cfg.seq_len_skip:cfg.seq_len_ctx+cfg.seq_len_skip+cfg.seq_len_n_in+100,:]
                # batch_u_dec = batch_u[:,cfg.seq_len_ctx+cfg.seq_len_skip:cfg.seq_len_ctx+cfg.seq_len_skip+cfg.seq_len_n_in+100,:]
                if device_type == "cuda":
                    batch_y_enc_fake = batch_y_enc_fake.pin_memory().to(device, non_blocking=True)
                    batch_u_enc_fake = batch_u_enc_fake.pin_memory().to(device, non_blocking=True)
                    batch_y_dec_fake = batch_y_dec_fake.pin_memory().to(device, non_blocking=True)
                    batch_u_dec_fake = batch_u_dec_fake.pin_memory().to(device, non_blocking=True)
                    _, std, _,_,prob = best_model(batch_y_enc_fake, batch_u_enc_fake, batch_u_dec_fake, batch_y_dec_fake,cfg.seq_len_n_in)
                    _, indices_2 = torch.topk(torch.exp(-prob).mean(axis = 1).view(-1),k = cfg.eval_batch_size, largest = False)
                    indices = indices_2[:32]
                    # remaining_indices = torch.tensor(indices_2[24:])
                    # indices =  torch.cat((indices_2[:24],remaining_indices[torch.randperm(len(remaining_indices))[:8]]))
                    batch_y_enc = batch_y_enc_fake[indices,:,:]
                    batch_u_enc = batch_u_enc_fake[indices,:,:]
                    batch_y_dec = batch_y_dec_fake[indices,:,:]
                    batch_u_dec = batch_u_dec_fake[indices,:,:]
                    batch_y_enc_2 = batch_y_enc_fake[indices_2,:,:]
                    batch_u_enc_2 = batch_u_enc_fake[indices_2,:,:]
                    batch_y_dec_2 = batch_y_dec_fake[indices_2,:,:]
                    batch_u_dec_2 = batch_u_dec_fake[indices_2,:,:]

                    _, _, loss_iter, rmse_iter,_ = model(batch_y_enc, batch_u_enc, batch_u_dec, batch_y_dec,cfg.seq_len_n_in)
                    _, _, loss_iter_2, rmse_iter_2,_ = model(batch_y_enc_2, batch_u_enc_2, batch_u_dec_2, batch_y_dec_2,cfg.seq_len_n_in)
                else:
                    _, _, loss_iter, rmse_iter,_ = model(batch_y_enc, batch_u_enc, batch_u_dec, batch_y_dec,cfg.seq_len_n_in)
                    _, _, loss_iter_2, rmse_iter_2,_ = model(batch_y_enc_2, batch_u_enc_2, batch_u_dec_2, batch_y_dec_2,cfg.seq_len_n_in)


                loss += loss_iter.item()
                rmse += rmse_iter.item()
                loss_2 += loss_iter_2.item()
                rmse_2 += rmse_iter_2.item()
                if eval_iter == cfg.eval_iters:
                    break
            loss /= cfg.eval_iters
            rmse/= cfg.eval_iters
            loss_2 /= cfg.eval_iters
            rmse_2/= cfg.eval_iters
            model.train()
            return loss, rmse, loss_2, rmse_2

        # Training loop
        LOSS_ITR = []
        LOSS_VAL = []
        SKIP_ITR = []
        loss_val = np.nan
        rmse_loss_val = np.nan
        loss_val_2 = np.nan
        rmse_loss_allval = np.nan

        if cfg.init_from == "scratch" or cfg.init_from == "pretrained":
            iter_num = 0
            iter_num_new =0
            best_val_loss = np.inf
            best_val_rmse = np.inf
        elif cfg.init_from == "resume":
            iter_num = checkpoint["iter_num"]
            iter_num_new = 0
            best_val_rmse = np.inf
            best_val_loss = np.inf#checkpoint['best_val_loss']

        get_lr = partial(warmup_cosine_lr, lr=cfg.lr, min_lr=cfg.min_lr,
                        warmup_iters=cfg.warmup_iters, lr_decay_iters=cfg.lr_decay_iters)
        iter_val = 0
        time_start = time.time()
        # time_limit = 60  # e.g., 10 minutes
        # max_iters = 500


        # Check if time limit has been exceeded
        exit_flag_finetune =0
        # Here instead you see a element and then Pass to the next realization of the dataset, Differnt from the validation!
        iteration_done = 0
        iter_cycle = 0
        for iter_num_new, (batch_y, batch_u) in tqdm.tqdm(enumerate(train_dl, start=iter_num_new)):
            

            # pass# print(iter_num)
            # print(batch_u.shape)
            
            
    

            # pass
            with torch.no_grad():
                batch_y_enc_fake = batch_y[:,:cfg.seq_len_ctx,:]
                batch_u_enc_fake = batch_u[:,:cfg.seq_len_ctx,:]
                batch_y_dec_fake = batch_y[:,cfg.seq_len_ctx+cfg.seq_len_skip:,:]
                batch_u_dec_fake = batch_u[:,cfg.seq_len_ctx+cfg.seq_len_skip:,:]
                if device_type == "cuda":
                    use_cuda = 1
                    batch_y_enc_fake = batch_y_enc_fake.pin_memory().to(device, non_blocking=True)
                    batch_u_enc_fake = batch_u_enc_fake.pin_memory().to(device, non_blocking=True)
                    batch_y_dec_fake = batch_y_dec_fake.pin_memory().to(device, non_blocking=True)
                    batch_u_dec_fake = batch_u_dec_fake.pin_memory().to(device, non_blocking=True)
                    _, std, _,_,prob = best_model(batch_y_enc_fake, batch_u_enc_fake, batch_u_dec_fake, batch_y_dec_fake,cfg.seq_len_n_in)
                    _, indices = torch.topk(torch.exp(-prob).mean(axis = 1).view(-1),k = cfg.batch_size, largest = False)
                    # if iter_num_new -iteration_done < iteration[iter_cycle]:
                    #     indices = indices[vec_choice[iter_cycle]-10:vec_choice[iter_cycle]]
                    #     indices = indices.cpu()
                    #     remaining_indices = torch.tensor(indices[vec_choice[iter_cycle]:])
                    # else:
                    #     iteration_done += iteration[iter_cycle]
                    #     iter_cycle += 1
                    #     # print(iter_cycle, iteration_done)
                    #     best_val_loss = np.inf
                    #     best_val_rmse = np.inf
                    #     indices = indices[vec_choice[iter_cycle]-10:vec_choice[iter_cycle]]
                    #     indices = indices.cpu()
                    # remaining_indices = torch.tensor(indices[1:])
                    # indices =  torch.cat((indices[:1],remaining_indices[torch.randperm(len(remaining_indices))[:31]]))
                    # indices =  indices[torch.randperm(len(indices))[:32]]
                    indices =  indices[:32]
                indices = indices.cpu()
                batch_y_enc_fake = batch_y_enc_fake.cpu()
                batch_u_enc_fake = batch_u_enc_fake.cpu()
                batch_y_dec_fake = batch_y_dec_fake.cpu()
                batch_u_dec_fake = batch_u_dec_fake.cpu()
                batch_y_enc = batch_y_enc_fake[indices,:,:]
                batch_u_enc = batch_u_enc_fake[indices,:,:]
                batch_y_dec = batch_y_dec_fake[indices,:,:]
                batch_u_dec = batch_u_dec_fake[indices,:,:]

                # for _,(b_y,b_u) in islice(enumerate(train_dl),1):
                #     with torch.no_grad():
                #         batch_y_enc_fake = b_y[:,:cfg.seq_len_ctx,:]
                #         batch_u_enc_fake = b_u[:,:cfg.seq_len_ctx,:]
                #         batch_y_dec_fake = b_y[:,cfg.seq_len_ctx+cfg.seq_len_skip:,:]
                #         batch_u_dec_fake = b_u[:,cfg.seq_len_ctx+cfg.seq_len_skip:,:]
                #         # if device_type == "cuda":
                #         batch_y_enc_fake = batch_y_enc_fake.pin_memory().to(device, non_blocking=True)
                #         batch_u_enc_fake = batch_u_enc_fake.pin_memory().to(device, non_blocking=True)
                #         batch_y_dec_fake = batch_y_dec_fake.pin_memory().to(device, non_blocking=True)
                #         batch_u_dec_fake = batch_u_dec_fake.pin_memory().to(device, non_blocking=True)
                #         _, _, _,_,prob = model(batch_y_enc_fake, batch_u_enc_fake, batch_u_dec_fake, batch_y_dec_fake,cfg.seq_len_n_in)
                #         _, indices = torch.topk(torch.exp(-prob).mean(axis = 1).view(-1),k = 80, largest = False)
                #         remaining_indices = torch.tensor(indices[8:])
                #         indices = torch.cat((indices[:8], remaining_indices[torch.randperm(len(remaining_indices))[:8]]))
                #     indices = indices.cpu()
                #     batch_y_enc_fake = batch_y_enc_fake.cpu()
                #     batch_u_enc_fake = batch_u_enc_fake.cpu()
                #     batch_y_dec_fake = batch_y_dec_fake.cpu()
                #     batch_u_dec_fake = batch_u_dec_fake.cpu()
                #     batch_y_enc = torch.cat((batch_y_enc,batch_y_enc_fake[indices,:,:]))
                #     batch_u_enc = torch.cat((batch_u_enc,batch_u_enc_fake[indices,:,:]))
                #     batch_y_dec = torch.cat((batch_y_dec,batch_y_dec_fake[indices,:,:]))
                #     batch_u_dec = torch.cat((batch_u_dec,batch_u_dec_fake[indices,:,:]))
                
            # if cfg.init_from == "resume" or cfg.init_from == "pretrained":
            #     # if time.time() - time_start > time_limit:
            #     if iter_num == checkpoint["iter_num"]:
            #         print("Time limit reached, stopping training.")
            #         break
            if ((iter_num+iter_num_new) % cfg.eval_interval == 0) and iter_num+iter_num_new > 0:# or iter_num == checkpoint["iter_num"]+1:
                print(f"\n loss val before {best_val_loss}")
                loss_val,rmse_loss_val,loss_val_2,rmse_loss_allval = estimate_loss(iter_cycle)
                LOSS_VAL.append(loss_val)
                print(f"\n{iter_num+iter_num_new=} {loss_val=:.4f},{rmse_loss_val=:.4f}\n")
                if loss_val < best_val_loss or rmse_loss_val< best_val_rmse:
                    print("changed.pt")
                    iter_val += 1
                    best_val_loss = loss_val
                    best_val_rmse = rmse_loss_val
                    # if ((iter_num+iter_num_new) % (cfg.eval_interval*4)== 0) and iter_num+iter_num_new > 0:# or iter_num == checkpoint["iter_num"]+1:
                    #     best_model = copy.deepcopy(model).to(device)
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num+iter_num_new,
                        'train_time': time.time() - time_start,
                        'LOSS': LOSS_ITR,
                        'LOSS_VAL': LOSS_VAL,
                        'best_val_loss': best_val_loss,
                        'cfg': cfg,
                    }
                    if cfg.log_wandb:
                        wandb.log({"iter_save": iter_val})
                    torch.save(checkpoint, model_dir / f"{cfg.out_file}.pt")
            # determine and set the learning rate for this iteration
            if cfg.decay_lr:
                lr_iter = get_lr(iter_num+iter_num_new)
            else:
                lr_iter = cfg.lr

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_iter
            
            
            # batch_y_dec = batch_y[:,cfg.seq_len_ctx+cfg.seq_len_skip:cfg.seq_len_ctx+cfg.seq_len_skip+cfg.seq_len_n_in+100,:]
            # batch_u_dec = batch_u[:,cfg.seq_len_ctx+cfg.seq_len_skip:cfg.seq_len_ctx+cfg.seq_len_skip+cfg.seq_len_n_in+100,:]
            if torch.isnan(batch_y_dec).any():
                print("Nan at iteration {iter_num}, skipping...") 
                SKIP_ITR.append(iter_num+iter_num_new)
                continue
            if device_type == "cuda":
                use_cuda = 1
                batch_y_enc = batch_y_enc.pin_memory().to(device, non_blocking=True)
                batch_u_enc = batch_u_enc.pin_memory().to(device, non_blocking=True)
                batch_y_dec = batch_y_dec.pin_memory().to(device, non_blocking=True)
                batch_u_dec = batch_u_dec.pin_memory().to(device, non_blocking=True)
                # _, _, _,_,prob = model(batch_y_enc, batch_u_enc, batch_u_dec, batch_y_dec,cfg.seq_len_n_in)
                # _, indices = torch.topk(torch.exp(-prob).mean(axis = 1).view(-1),k = 32, largest = False)

                # batch_y_enc = batch_y_enc[indices,:,:]
                # batch_u_enc = batch_u_enc[indices,:,:]
                # batch_y_dec = batch_y_dec[indices,:,:]
                # batch_u_dec = batch_u_dec[indices,:,:]
                _, _, loss,rmse_loss,_ = model(batch_y_enc, batch_u_enc, batch_u_dec, batch_y_dec,cfg.seq_len_n_in)
            else:
                _, _, loss,rmse_loss,_ = model(batch_y_enc, batch_u_enc, batch_u_dec, batch_y_dec,cfg.seq_len_n_in)

            LOSS_ITR.append(loss.item())
            if (iter_num+iter_num_new) % 100 == 0 : 
                print(f"\n{iter_num+iter_num_new=} {loss=:.4f} {loss_val=:.4f} {rmse_loss=:.4f} {rmse_loss_val=:.4f}  {rmse_loss_allval=:.4f} {lr_iter=}\n")
                if cfg.log_wandb:
                    wandb.log({"loss": loss, "loss_val": loss_val, "RMSE_val": rmse_loss_val,"RMSE_all_val": rmse_loss_allval})


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iter_num_new == max_iter-1:
                break
        
    except KeyboardInterrupt:
        print("closing gracefully")
        sys.exit() 

    time_loop = time.time() - time_start
    print(f"\n{time_loop=:.2f} seconds.")

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num+iter_num_new,
        'train_time': time.time() - time_start,
        'LOSS': LOSS_ITR,
        'LOSS_VAL': LOSS_VAL,
        'best_val_loss': best_val_loss,
        'cfg': cfg,
    }
    
    torch.save(checkpoint, model_dir / f"{cfg.out_file}_last.pt")

    if cfg.log_wandb:
        wandb.finish()

