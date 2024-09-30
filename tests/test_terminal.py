import torch
from tqdm import tqdm
import argparse
from intention_learning.terminal import TerminalModel, TerminalNetwork
from intention_learning.data import DataHandler, JudgmentBuffer, StateBuffer, IMAGES_DIR
from intention_learning.judge import Judge, validate_judgments
from intention_learning.img import ImageHandler

def main():
    parser = argparse.ArgumentParser(description="Test Terminal Model")
    parser.add_argument('--id', type=str, required=True, help='Identifier for the model and data')
    parser.add_argument('--action', type=str, choices=['judge-train', 'train', 'visualize'], help='Action to perform: judge-train, train, or visualize the terminal model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_buffer = StateBuffer(max_size=int(1e7), device=device)
    image_handler = ImageHandler()

    if args.action == "train":
        id = args.id
        judgment_buffer = DataHandler.load_buffer(id, JudgmentBuffer, device)
        judgments = judgment_buffer.all()
        judgment_values = judgments["judgments"]
        print(f"{judgment_values.shape=}")
        print(f"{torch.mean(judgment_values)=}")
        print(torch.sum(judgment_values) / judgment_values.numel())
        is_valid_judgment = validate_judgments(**judgments)
        print(f"percent valid: {is_valid_judgment.sum() / is_valid_judgment.numel()}")
        data_handler = DataHandler(state_buffer, judgment_buffer)
        terminal_model = TerminalModel(data_handler, device)

        print(f"{judgment_buffer.size=}")
        print(f"{terminal_model.sample_and_evaluate_loss(n_samples=int(1e4))=}")
        terminal_model.train(n_iter=int(1e4), batch_size=int(1e4))
        print(f"{terminal_model.sample_and_evaluate_loss(n_samples=int(1e4))=}")
    elif args.action == "judge-train":
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

            # get accuracy of judgments
            judgments = judgment_buffer.all()
            is_valid_judgment = validate_judgments(**judgments)
            percent_valid = is_valid_judgment.sum() / is_valid_judgment.numel()

            # add loss to progress bar
            loss = terminal_model.sample_and_evaluate_loss(n_samples=terminal_train_batch_size)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Judgments": judgment_buffer.size, "Percent Valid": f"{percent_valid:.4f}"})

            if epoch % 5 == 0:
                terminal_model.save()
                data_handler.save_buffer(judgment_buffer)
    elif args.action == "visualize":
        # Load the terminal model with the specified ID
        judgment_buffer = DataHandler.load_buffer(args.id, JudgmentBuffer, device)
        print(f"{judgment_buffer.size=}")
        is_valid_judgment = validate_judgments(**judgment_buffer.all())
        percent_valid = is_valid_judgment.sum() / is_valid_judgment.numel()
        print(f"percent valid: {percent_valid:.4f}")
        data_handler = DataHandler(state_buffer, judgment_buffer)
        terminal_network = DataHandler.load_model(args.id, TerminalNetwork)
        terminal_network = terminal_network.to(device)
        terminal_model = TerminalModel(data_handler, device, network=terminal_network)

        # Generate the visualization
        visualization = image_handler.visualize_terminal_model(terminal_model, device=device)

        # Save the visualization
        visualization_path = IMAGES_DIR / f"terminal_model_visualization_{args.id}.png"
        visualization.save(visualization_path)
        print(f"Visualization saved at {visualization_path}")

if __name__ == "__main__":
    main()
