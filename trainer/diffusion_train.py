import time
import torch
from data.diffusion_dataset import Sampler, u, r
from nn.pde import diffusion_operator
def fetch_minibatch(sampler, N):
    X, Y = sampler.sample(N)
    return X, Y
def train(model, nIter=10000, batch_size=128, log_NTK=False, update_lam=False):
    ics_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=torch.float32, device=model.device
    )
    bc1_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float32, device=model.device
    )
    bc2_coords = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=model.device
    )
    dom_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=model.device
    )
    ics_sampler = Sampler(3, ics_coords, u, name="Initial Condition", device=model.device)
    bc1 = Sampler(3, bc1_coords, u, name="Dirichlet BC1", device=model.device)
    bc2 = Sampler(3, bc2_coords, u, name="Dirichlet BC2", device=model.device)
    bcs_sampler = [bc1, bc2]
    res_sampler = Sampler(3, dom_coords, r, name="Forcing", device=model.device)
    training_start_time = time.time()
    model.logger.print(f"Starting training for {model.epochs} epochs...")
    model.logger.print(f"Batch size: {batch_size}")
    
    def objective_fn(it):
        start_time = time.time()
        if model.optimizer is not None:
            model.optimizer.zero_grad()
        X_ics_batch, u_ics_batch = fetch_minibatch(ics_sampler, batch_size // 3)
        X_bcs_batch, u_bcs_batch = fetch_minibatch(bcs_sampler[0], batch_size // 3)
        X_res_batch, r_res_batch = fetch_minibatch(res_sampler, batch_size)
        X_ics_batch.requires_grad_(True)
        t_ics = X_ics_batch[:, 0:1]  
        t_ics.requires_grad_(True)
        u_bc1_pred = model.forward(X_bcs_batch)
        u_ics_pred = model.forward(X_ics_batch)
        t_r, x_r, y_r = X_res_batch[:, 0:1], X_res_batch[:, 1:2], X_res_batch[:, 2:3]
        [_, r_pred] = diffusion_operator(model, t_r, x_r, y_r)
        loss_r = model.loss_fn(r_pred, r_res_batch)
        loss_bc1 = model.loss_fn(u_bc1_pred, u_bcs_batch)
        loss_ics = model.loss_fn(u_ics_pred, u_ics_batch)
        loss = 2.0 * (loss_r) + 4.0 * loss_bc1 + 2.0 * loss_ics
        elapsed = time.time() - start_time
        return loss, elapsed, loss_r, loss_bc1, loss_ics
    
    epoch_times = []
    for it in range(model.epochs + 1):
        loss, epoch_time, loss_r, loss_bc1, loss_ics = objective_fn(it)
        epoch_times.append(epoch_time)
        
        if it % model.args["print_every"] == 0 or it == 0 or model.args.get("use_ibm_hardware", False):
            total_elapsed = time.time() - training_start_time
            avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
            remaining_epochs = model.epochs - it
            eta_seconds = avg_epoch_time * remaining_epochs if avg_epoch_time > 0 else 0
            
            model.logger.print(
                "Epoch: %d/%d [%.1f%%] | Loss: %.2e | Loss_res: %.2e | Loss_bcs: %.2e | loss_ics: %.2e | lr: %.2e | Epoch_time: %.2fs | Total: %.1fs | ETA: %.1fs"
                % (
                    it,
                    model.epochs,
                    100.0 * it / model.epochs if model.epochs > 0 else 0,
                    loss.item(),
                    loss_r.item(),
                    loss_bc1.item(),
                    loss_ics.item(),
                    model.optimizer.param_groups[0]["lr"] if model.optimizer else 0.0,
                    epoch_time,
                    total_elapsed,
                    eta_seconds,
                )
            )
            if it > 0 and it % model.args["print_every"] == 0:
                model.save_state()
        
        loss.backward()
        if model.args["solver"] == "CV":
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        if model.optimizer is not None:
            model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.step(loss)  
        model.loss_history.append(loss.item())
    
    total_training_time = time.time() - training_start_time
    model.logger.print(f"Training completed in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")