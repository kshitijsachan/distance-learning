import pickle, gzip, argparse, ipdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str)
    args = parser.parse_args()

    traj_lengths = []
    with gzip.open(args.file_name, 'rb') as f:
        try:
            while True:
                traj = pickle.load(f)
                traj_lengths.append(len(traj))
        except EOFError:
            print(f"{args.file_name} contains {len(traj_lengths)} trajectories")
        

