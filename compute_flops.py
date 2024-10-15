import argparse
from tqdm import tqdm
import torch
from model_farsecnn import LitFARSECNN
from model_detect_farsecnn import LitDetectFARSECNN
from utils.data_utils import get_dataset

def compute_network_flops(net, dataloader, device = torch.device('cpu')):
    flops_per_layer = [0] * len(net.farsecnn)
    n_ev_per_layer = [0] * len(net.farsecnn)

    for batch in tqdm(iter(dataloader)):
        batch = {k: t.to(device) for k, t in batch.items()}
        events, lengths = batch['events'], batch['lengths']

        x = net.preprocess_inputs(events, lengths)

        with torch.no_grad():
            for i, l in enumerate(net.farsecnn):
                n_ev_per_layer[i] = n_ev_per_layer[i] + x['events'].shape[0]
                flops_per_layer[i] = flops_per_layer[i] + l.compute_flops(x)
                x = l(x)

    return flops_per_layer, n_ev_per_layer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg",
                        help="Path to model config file.")
    parser.add_argument("--dataset",
                        help="Name of the dataset to compute the FLOPS.")
    parser.add_argument("--bs", default=1, type=int,
                        help="Batch size to use for computation.")
    args = parser.parse_args()

    model_cfg = args.model_cfg
    dataset_name = args.dataset
    bs = args.bs

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Available device %s" % device)

    print(f"Building model from config: %s" % model_cfg)
    if 'Gen1' in dataset_name:
        net = LitDetectFARSECNN(model_cfg, bs=bs, dataset=dataset_name).to(device)
    else:
        net = LitFARSECNN(model_cfg, bs=bs, dataset=dataset_name).to(device)

    import transforms
    transforms = transforms.UniformNoise(n=33981, use_range=False)
    _, _, dataset = get_dataset(dataset_name, test_transform=transforms) # FLOPS are computed on the test dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=net.pad_batches)
    print(f"Testing on dataset: %s" % dataset)

    print("Computing FLOPS...")
    flops_per_layer, n_ev_per_layer = compute_network_flops(net, dataloader, device)

    in_events = n_ev_per_layer[0]

    print(f"Total FLOPS:\t %d" % sum(flops_per_layer))
    print(f"Total FLOPS/ev:\t %.3f M" % (sum(flops_per_layer)/in_events/1e6))

    for i, (f,ev) in enumerate(zip(flops_per_layer,n_ev_per_layer)):
        print(f"%s\t\tFLOPS/ev:%.3f M\t#events received:%d" % (net.farsecnn[i].__class__.__name__, (f/in_events/1e6), ev))