import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="mycli",
        description="My awesome command line tool"
    )

    parser.add_argument("name", help="Your name")
    parser.add_argument(
        "--greeting",
        default="Hello",
        help="Greeting to use"
    )

    args = parser.parse_args()

    print(f"{args.greeting}, {args.name}!")


if __name__ == "__main__":
    main()