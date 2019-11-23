import jieba 
import pandas
import codecs
import re 
def subReplace(line):
    regex = re.compile(u'[^\u4e00-\u9fa5.，,。？“”]+', re.UNICODE)
    return regex.sub('', line)
def build_vocab(train_csv, test_csv, vocab_path):
    vocab = set()
    df = pandas.read_csv(train_csv)
    test_df = pandas.read_csv(test_csv)
    # for i in range(len(df)):
    for i in range(len(df)):
        text = df.iloc[i]['text']
        # remove space
        text = text.replace(' ', '')
        text = subReplace(text)
        # cut chinese word
        seg_list = list(jieba.cut(text))
        for word in seg_list:
            vocab.add(word)
        df.at[i, 'text'] = '/'.join(seg_list)
        
        
        
        # print(df.iloc[i]['text'])
    df.to_csv('data/train_set.csv', index=0)

    for i in range(len(test_df)):
        text = test_df.iloc[i]['text']
        text = text.replace(' ', '')
        text = subReplace(text)
        seg_list = list(jieba.cut(text))
        test_df.at[i, 'text'] = '/'.join(seg_list)
    test_df.to_csv('data/test_set.csv', index=0)
    vocab.add('UNK')
    vocab.add('PAD')

    
    with codecs.open(vocab_path, 'w', 'utf-8') as vocab_file:
        for key in vocab:
            vocab_file.write(key + "\n")

if __name__ == "__main__":
    build_vocab('../THUNews_frag/train_set.csv', '../THUNews_frag/test_set.csv', 'data/vocab.txt')
    

