import argparse
from util import create_tfrecords
def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]
    parser = argparse.ArgumentParser(description='Bash model training script')
    parser.add_argument('--save_path', type=str, default='.',
                        help='Path to save TFRecords files to.')
    parser.add_argument('--min_length', type=int, default=2,
                        help='Filter data to only include sequences with '
                        'length >min_length')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Shuffle samples uniformly randomly as we write')
    parser.add_argument('--single_out_user', type=int, default=-1,
                        help='Creates a dataset with example labels replaced '
                        'with (equals single_out_user) and '
                        '(not equals single_out_user)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Ratio of test set to total dataset')
    return parser.parse_args()

def main():
    args = get_arguments()
    create_tfrecords(save_path=args.save_path, 
                     shuffle=args.shuffle, 
                     single_out_user=args.single_out_user, 
                     min_length=args.min_length,
                     test_size=args.test_size)

if __name__ == '__main__':
    main()
    