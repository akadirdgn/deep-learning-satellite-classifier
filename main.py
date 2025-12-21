import argparse
import sys
from scripts.train_pipeline import train
from scripts.evaluate_model import evaluate
from scripts.prepare_dataset import prepare

# Windows unicode cikti sorunu icin (Gerek kalmadi, emojileri kaldirdik)
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def main():
    parser = argparse.ArgumentParser(description="Uydu Goruntu Siniflandirma Projesi")
    parser.add_argument('mode', choices=['train', 'evaluate', 'prepare'], help="Calistirma modu: 'train', 'evaluate' veya 'prepare'")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        evaluate()
    elif args.mode == 'prepare':
        prepare()

if __name__ == "__main__":
    main()
