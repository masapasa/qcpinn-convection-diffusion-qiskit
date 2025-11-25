import torch
import os
import sys
import matplotlib.pyplot as plt 
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logging
from nn.pde import diffusion_operator
from utils.ContourPlotter import ContourPlotter
from data.diffusion_dataset import u, r
import trainer.diffusion_train as diffusion_train
from nn.DVPDESolver import DVPDESolver
from nn.CVPDESolver import CVPDESolver
from nn.ClassicalSolver import ClassicalSolver

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IBM QUANTUM SETTINGS
# Set USE_IBM_HARDWARE to True to use Qiskit Runtime + Real Hardware
USE_IBM_HARDWARE = True 
# Replace with your actual API Token

IBM_TOKEN = "rTMZyypbjUUZ9Q_jjydgC-HAkVb2OJd42YAmIRtrDAvl"
# Instance CRN (optional). If None, uses default instance.
# Format: "crn:v1:bluemix:public:quantum-computing:region:a/account:instance_id::" "crn:v1:bluemix:public:quantum-computing:us-east:a/e6c5d7bd99fc4557ab273e4a7000a964:5bbab335-aee1-475d-a97c-33cfb67d80d3::")
IBM_INSTANCE = None
# Backend name. If not found, will auto-select from available backends.
# Examples: 'ibm_torino', 'ibm_fez', 'ibm_marrakesh'
IBM_BACKEND_NAME = "ibmq_qasm_simulator" 

mode = "hybrid"
num_qubits = 4
output_dim = 1
input_dim = 3
hidden_dim = 32
num_quantum_layers = 1
cutoff_dim = 20
classic_network = [input_dim, hidden_dim, output_dim]

args = {
    "batch_size": 32,
    "epochs": 50,
    "lr": 0.005,
    "seed": 1,
    "print_every": 10,
    "log_path": "./checkpoints/diffusion",
    "input_dim": input_dim,
    "output_dim": output_dim,
    "num_qubits": num_qubits,
    "hidden_dim": hidden_dim,
    "num_quantum_layers": num_quantum_layers,
    "classic_network": classic_network,
    "q_ansatz": "cascade",  
    "mode": mode,
    "activation": "tanh",  
    "shots": 256,
    "problem": "diffusion",
    "solver": "DV", # Must be DV (Discrete Variable) for Qiskit/IBM Hardware
    "device": DEVICE,
    "method": "None",
    "cutoff_dim": cutoff_dim,  
    "class": "CVNeuralNetwork1",  
    "encoding": "None",
    
    # IBM Hardware Args
    "use_ibm_hardware": USE_IBM_HARDWARE,
    "ibm_token": IBM_TOKEN,
    "ibm_backend": IBM_BACKEND_NAME,
    "ibm_instance": IBM_INSTANCE
}

log_path = args["log_path"]
logger = Logging(log_path)

# Model Initialization Logic
if args["solver"] == "CV":
    # Continuous Variable (Photonic) - Not compatible with IBM Qiskit (Qubit-based)
    if USE_IBM_HARDWARE:
        raise ValueError("CV Solver (Photonic) cannot run on IBM Quantum Hardware (Qubit-based). Set solver to 'DV'.")
    model = CVPDESolver(args, logger, DEVICE)
    model.logger.print("Using CV Solver")
elif args["solver"] == "Classical":
    model = ClassicalSolver(args, logger, DEVICE)
    model.logger.print("Using Classical Solver")
else:
    # Discrete Variable (Qubit) - Compatible with IBM Qiskit
    model = DVPDESolver(args, logger, DEVICE)
    model.logger.print("Using DV Solver")
    if USE_IBM_HARDWARE:
        model.logger.print(f"--> CONNECTED TO IBM QUANTUM BACKEND: {IBM_BACKEND_NAME}")

model.logger.print(f"The settings used:")
for key, value in args.items():
    # Mask token in logs
    if key == "ibm_token":
        model.logger.print(f"{key} : ********************")
    else:
        model.logger.print(f"{key} : {value}")

total_params = sum(p.numel() for p in model.parameters())
model.logger.print(f"Total number of parameters: {total_params}")

# --- TRAINING ---
diffusion_train.train(model)
model.save_state()
model.logger.print("Training completed successfuly!")

# --- PLOTTING LOSS ---
plt.plot(range(len(model.loss_history)), model.loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()
file_path = os.path.join(model.log_path, "loss_history.pdf")
plt.savefig(file_path, bbox_inches="tight")
plt.show()
plt.close("all")

model.logger.print(f"The last loss is: , {model.loss_history[-1]}")

# --- EVALUATION ---
NUM_OF_POINTS = 10
dom_coords = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=DEVICE
)

time_ = (
    torch.linspace(dom_coords[0, 0], dom_coords[1, 0], NUM_OF_POINTS)
    .to(DEVICE)
    .unsqueeze(1)
    .to(torch.float32)
)
xfa = (
    torch.linspace(dom_coords[0, 1], dom_coords[1, 1], NUM_OF_POINTS)
    .to(DEVICE)
    .unsqueeze(1)
    .to(torch.float32)
)
yfa = (
    torch.linspace(dom_coords[0, 2], dom_coords[1, 2], NUM_OF_POINTS)
    .to(DEVICE)
    .unsqueeze(1)
    .to(torch.float32)
)

time_, xfa, yfa = torch.meshgrid(time_.squeeze(), xfa.squeeze(), yfa.squeeze(), indexing='ij')

X_star = torch.hstack(
    (
        time_.flatten().unsqueeze(1),
        xfa.flatten().unsqueeze(1),
        yfa.flatten().unsqueeze(1),
    )
).to(DEVICE)

# Inference
u_pred, f_pred = diffusion_operator(
    model, X_star[:, 0:1], X_star[:, 1:2], X_star[:, 2:3]
)

if u_pred.is_cuda:
    u_pred = u_pred.cpu()
    f_pred = f_pred.cpu()

u_pred = u_pred.detach().numpy()
f_pred = f_pred.detach().numpy()

u_analytic = u(X_star).cpu().detach().numpy()
f_analytic = r(X_star).cpu().detach().numpy()

# Error Calculation
error_u = (
    np.linalg.norm(u_analytic - u_pred, 2) / np.linalg.norm(u_analytic, 2)
) * 100.0
error_f = (
    np.linalg.norm(f_analytic - f_pred, 2) / np.linalg.norm(f_analytic + 1e-9, 2)
) * 100.0

logger.print("Relative L2 error_u: {:.2e}".format(error_u))
logger.print("Relative L2 error_f: {:.2e}".format(error_f))

# --- CONTOUR PLOTTING ---
tstep = NUM_OF_POINTS
xstep = NUM_OF_POINTS
ystep = NUM_OF_POINTS

X = X_star.cpu().detach().numpy()
exact_velocity = u_analytic
exact_force = f_analytic

xf = xfa.reshape(tstep, xstep, ystep).cpu().detach().numpy()  
yf = yfa.reshape(tstep, xstep, ystep).cpu().detach().numpy()  

exact_velocity = exact_velocity.reshape(tstep, xstep, ystep)  
exact_force = exact_force.reshape(tstep, xstep, ystep)  
grbf_velocity = u_pred.reshape(tstep, xstep, ystep)  
grbf_force = f_pred.reshape(tstep, xstep, ystep)  

titles = [
    "exact_u",
    "exact_p",
    "pred_u_classic",
    "pred_p_classic",
    "abs_error_u_classic",
    "abs_error_p_classic",
]

nrows_ncols = (3, 2)
values = [99]
xref = 1
yref = 1
model_dirname = model.log_path
img_width = 10
img_height = 10
ticks = 3
fontsize = 7
labelsize = 7
axes_pad = 0.5

visualization_data = [
    exact_velocity,  
    exact_force,  
    grbf_velocity,  
    grbf_force,  
    np.abs(exact_velocity - grbf_velocity),  
    np.abs(exact_force - grbf_force),  
]

plotter = ContourPlotter(fontsize=7, labelsize=7, axes_pad=0.5)

plotter.draw_contourf_regular_2D(
    time_[:, 0, 0].cpu().detach().numpy(), # Ensure cpu numpy for plotting
    xf[0, :, 0],
    yf[0, 0, :],
    visualization_data,
    titles=titles,
    nrows_ncols=nrows_ncols,
    time_steps=[10],
    xref=1,
    yref=1,
    model_dirname=model_dirname,
    img_width=10,
    img_height=10,
    ticks=3,
)