import torch
torch.set_float32_matmul_precision('high')

from comet import download_model, load_from_checkpoint

def main():
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)
    data = [
        {
            "src": "Boris Johnson teeters on edge of favour with Tory MPs", 
            "mt": "Boris Johnson ist bei Tory-Abgeordneten völlig in der Gunst", 
            "ref": "Boris Johnsons Beliebtheit bei Tory-MPs steht auf der Kippe"
        }
    ]
    print(data)

    model_output = model.predict(data, batch_size=8, gpus=1)
    # Segment-level scores
    print (model_output.scores)

    # System-level score
    print (model_output.system_score)

    # Score explanation (error spans)
    print (model_output.metadata.error_spans)

if __name__ == "__main__":
    main()