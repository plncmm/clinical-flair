from flair.embeddings import StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from typing import List

class ContextualEmbeddings:
    def __init__(self, model, domain) -> None:
        self.type = model
        self.domain = domain
    
    def create_embeddings(self) -> StackedEmbeddings:
        embedding_types: List[FlairEmbeddings] = []
        if self.type == 'flair':
            

            # Our domain-specific Flair embeddings
            if self.domain == 'clinical':
                embedding_types.append(FlairEmbeddings('es-clinical-forward'))
                embedding_types.append(FlairEmbeddings('es-clinical-backward'))

            if self.domain == 'scielo':
                embedding_types.append(FlairEmbeddings('../scielo-forward.pt'))
                embedding_types.append(FlairEmbeddings('../scielo-backward.pt'))

            # Base general-domain flair embeddings used to train our model
            if self.domain == 'general-domain':
                embedding_types.append(FlairEmbeddings('es-forward'))
                embedding_types.append(FlairEmbeddings('es-backward'))

            embeddings: StackedEmbeddings = StackedEmbeddings(embeddings = embedding_types)
        
        if self.type == 'bert':
            if self.domain == 'clinical':
                embedding_types.append(TransformerWordEmbeddings(
                'PlanTL-GOB-ES/bsc-bio-ehr-es', 
                layers = 'all', 
                layer_mean = True, 
                subtoken_pooling = 'first'
            ))
    

            # Base general-domain flair embeddings used to train our model
            if self.domain == 'general-domain':
                embedding_types.append(TransformerWordEmbeddings(
                'dccuchile/bert-base-spanish-wwm-cased', 
                layers = 'all', 
                layer_mean = True, 
                subtoken_pooling = 'first'
            ))

            embeddings: StackedEmbeddings = StackedEmbeddings(embeddings = embedding_types)
            pass
        
        return embeddings