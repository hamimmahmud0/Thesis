import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="stabilize",
        description="Stabilize Drone footage for trajectory extraction"
    )

    parser.add_argument("input", help="input.mp4")
    parser.add_argument(
        "--backbone",
        default="cotracker",
        help="Select on of this backbone: cotracker"
    )

    args = parser.parse_args()

    if args.input == "cotracker":
        import stabilize.cotracker as cotracker
        


if __name__ == "__main__":
    main()