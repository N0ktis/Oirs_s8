import click
from preprocessing import get_prediction
import os.path as pth


@click.command()
@click.argument('file_path', type=str)
@click.option('--model_path', '-m', default='/models_data/model.pt', type=str)
@click.option('--dict_path', '-d', default='/models_data/tfidf.pkl', type=str)
@click.option('--pca_path', '-d', default='/models_data/pca.pkl', type=str)
@click.option('--threshold', '-t', default=3.5, type=float)
def main(file_path, model_path, dict_path, pca_path, threshold):
    for file, t_file in zip((file_path, model_path, dict_path, pca_path), ('.csv', '.pt', '.pkl', '.pkl')):
        if not (pth.isfile(file) and pth.splitext(file)[1] == t_file):
            raise Exception("File {file} does not exist or invalid type of file. Required {t_file}".format(file=file,
                                                                                                           t_file=t_file))
    filename = get_prediction(file_path, model_path, dict_path, pca_path, threshold)
    click.echo('{file} has been created'.format(file=filename))


if __name__ == '__main__':
    main()
