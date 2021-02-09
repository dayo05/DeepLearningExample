import urllib.request
import pandas as pd


urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="data/IMDb_Reviews.csv")
print('Download Finished')
df = pd.read_csv('data/IMDb_Reviews.csv', encoding='latin1')
print(df.head())

print('Count of sample is {}'.format(len(df)))

train_df = df[:25000]
test_df = df[25000:]

train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)

from torchtext import data

# Declare field
TEXT = data.Field(
    sequential=True,
    use_vocab=True,
    tokenize=str.split,
    lower=True,
    batch_first=True,
    fix_length=20
)

LABEL = data.Field(
    sequential=False,
    use_vocab=False,
    batch_first=False,
    is_target=True
)

from torchtext.data import TabularDataset


train_data, test_data = TabularDataset.splits(
    path='.',
    train='data/train_data.csv',
    test='data/test_data.csv',
    format='csv',
    fields=[('text', TEXT), ('label', LABEL)],
    skip_header=True
)

print('Count of train data: {}'.format(len(train_data)))
print('Count of test data: {}'.format(len(test_data)))
print(vars(train_data[0]))

# Display build of field
print(train_data.fields.items())

TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
print('Count of vocab: {}'.format(len(TEXT.vocab)))
print(TEXT.vocab.stoi)

from torchtext.data import Iterator


batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size=batch_size)
test_loader = Iterator(dataset=test_data, batch_size=batch_size)
print('Count of mini batch of train data: {}'.format(len(train_loader)))
print('Count of mini batch of test data: {}'.format(len(test_loader)))

# First mini batch
batch = next(iter(train_loader))
print(type(batch))
print(batch.text)
