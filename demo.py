from classification_multi import classification_multi
from args import parameter_parser

if __name__ == '__main__':
    args = parameter_parser()
    for args.dataset in ['ACM3025_0', 'DBLP_0', 'imdb5k_0', 'yelp_0']:
        print('--------------Multi-relational Datasets: {}--------------------'.format(args.dataset))
        classification_multi(args)
    for args.dataset in ['MNIST', 'HW', 'animals']:
        print('--------------Multi-attribute Datasets: {}--------------------'.format(args.dataset))
        classification_multi(args)
    for args.dataset in ['BDGP', 'esp-game', 'flickr']:
        print('--------------Multi-modality Datasets: {}--------------------'.format(args.dataset))
        classification_multi(args)