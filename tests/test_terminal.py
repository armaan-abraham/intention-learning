import torch
from tqdm import tqdm
from intention_learning.terminal import TerminalModel
from intention_learning.data import DataHandler, JudgmentBuffer, StateBuffer
from intention_learning.judge import Judge, validate_judgments
from intention_learning.img import ImageHandler

# Generate random states and store them with data manager
# Judge and store judgments
# Train terminal model

LOAD_JUDGMENTS = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_buffer = StateBuffer(max_size=int(1e7), device=device)
image_handler = ImageHandler()

if LOAD_JUDGMENTS:
    id = "e40be892"
    judgment_buffer = DataHandler.load_buffer(id, JudgmentBuffer, device)
    judgments = judgment_buffer.all()
    judgment_values = judgments["judgments"]
    print(torch.sum(judgment_values) / judgment_values.numel())
    is_valid_judgment = validate_judgments(**judgments)
    print(f"percent valid: {is_valid_judgment.sum() / is_valid_judgment.numel()}")
    data_handler = DataHandler(state_buffer, judgment_buffer)
    terminal_model = TerminalModel(data_handler, device)

    print(f"{judgment_buffer.size=}")
    print(f"{terminal_model.sample_and_evaluate_loss(n_samples=int(1e4))=}")
    terminal_model.train(n_iter=int(1e4), batch_size=int(1e4))
    print(f"{terminal_model.sample_and_evaluate_loss(n_samples=int(1e4))=}")
else:
    judgment_buffer = JudgmentBuffer(max_size=int(1e7), device=device)
    data_handler = DataHandler(state_buffer, judgment_buffer)
    judge = Judge(image_handler, data_handler)
    terminal_model = TerminalModel(data_handler, device)

    # store all random states
    n_states = 10000
    # uniform distribution over [0, 1]
    angles = torch.rand(n_states) * 2 * torch.pi
    velocities = torch.rand(n_states) * 16 - 8
    states = torch.stack([torch.cos(angles), torch.sin(angles), velocities], dim=1)
    data_handler.store_states(states)

    n_epochs = 100
    n_judgments_per_epoch = 200
    judgment_batch_size = 50
    terminal_train_batch_size = 200
    terminal_train_n_iter = 20

    # create progress bar
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        judge.sample_and_judge(n_pairs=n_judgments_per_epoch, batch_size=judgment_batch_size)
        terminal_model.train(batch_size=terminal_train_batch_size, n_iter=terminal_train_n_iter)

        # add loss to progress bar
        loss = terminal_model.sample_and_evaluate_loss(n_samples=terminal_train_batch_size)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Judgments": judgment_buffer.size})

        # save every 10 epochs
        if epoch % 10 == 0:
            terminal_model.save()
            data_handler.save_buffer(judgment_buffer, "judgment")
