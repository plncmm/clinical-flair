from unicodedata import bidirectional
from datasets import NERCorpus
from models import NERTagger
from embeddings import ContextualEmbeddings
from trainers import NERTrainer
import yaml
import torch 
import flair



if __name__=='__main__':

    # Read configuration file
    with open('../config.yaml') as file:
        config = yaml.safe_load(file)


    # Device
    available_gpu = torch.cuda.is_available()
    if available_gpu:
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        flair.device = torch.device('cuda')
    else:
        flair.device = torch.device('cpu')

    flair.set_seed(123)

    # Create corpus
    corpus_name = config['dataset']
    task = ''
    if corpus_name in ('cantemist', 'nubes', 'pharmaconer', 'clinical_trials'):
        corpus = NERCorpus(corpus_name).create_corpus()
        print(corpus)
        # print the first Sentence in the training split
        label_type = 'ner'

        # 3. make the label dictionary from the corpus
        label_dict = corpus.make_label_dictionary(label_type=label_type)
        
        # Embeddings
        embeddings = ContextualEmbeddings(config['model'], config['domain']).create_embeddings()
        
        
        # Create Sequence Labeling Model
        tagger = NERTagger(embeddings = embeddings,
            encoder = config['ner_hyperparameters']['encoder'],
            encoder_layers = config['ner_hyperparameters']['encoder_layers'],
            use_crf = config['ner_hyperparameters']['use_crf'],
            hidden_size = config['ner_hyperparameters']['hidden_size'],
            dropout = config['ner_hyperparameters']['dropout'],
            tag_dictionary=label_dict
        ).create_tagger()

        # Create Sequence Labeling Trainer
        trainer = NERTrainer(corpus = corpus,
            tagger = tagger,
            epochs = config['ner_training']['max_epochs'],
            learning_rate = config['ner_training']['learning_rate'],
            mini_batch_size = config['ner_training']['mini_batch_size'],
            optimizer = config['ner_training']['optimizer'],
            output_path = '{}_output_seed_123_roberta/'.format(corpus_name)
        ).train()

