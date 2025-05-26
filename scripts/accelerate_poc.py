from accelerate import PartialState
import torch


def main():
    state = PartialState()

    with state.split_between_processes([1, 2, 3]) as job:
        print(state.device, job, torch.cuda.get_device_capability(state.device))


if __name__ == "__main__":
    main()
