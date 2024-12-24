from datasets import load_dataset

# dataset = load_dataset('banking77')

# train = load_dataset('banking77', split = 'train')
# test = load_dataset('banking77', split = 'test')

# train.to_csv('banking77/train.tsv', sep = '\t')
# test.to_csv('banking77/dev.tsv', sep = '\t')

print('Getting IMDB')
dataset = load_dataset('stanfordnlp/imdb')

train = load_dataset('stanfordnlp/imdb', split = 'train')
test = load_dataset('stanfordnlp/imdb', split = 'test')

train.to_csv('imdb/train.tsv', sep = '\t')
test.to_csv('imdb/dev.tsv', sep = '\t')
print('Finished IMDB')

print('Getting sst5')
dataset = load_dataset('SetFit/sst5')

train = load_dataset('SetFit/sst5', split = 'train')
test = load_dataset('SetFit/sst5', split = 'test')

train.to_csv('sst5/train.tsv', sep = '\t')
test.to_csv('sst5/dev.tsv', sep = '\t')
print('Finished sst5')

print('Getting sst2')
dataset = load_dataset('SetFit/sst2')

train = load_dataset('SetFit/sst2', split = 'train')
test = load_dataset('SetFit/sst2', split = 'test')

train.to_csv('sst2/train.tsv', sep = '\t')
test.to_csv('sst2/dev.tsv', sep = '\t')
print('Finished sst2')


