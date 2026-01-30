from datasets import load_dataset

def main():
    print("Loading jailbreak dataset...")
    ds = load_dataset("BallAdMyFi/jailbreaking_prompt_v2")

    print("\nDataset structure:")
    print(ds)

    print("\nSample record:")
    print(ds["train"][0])

    print("\nColumn names:")
    print(ds["train"].column_names)

if __name__ == "__main__":
    main()
