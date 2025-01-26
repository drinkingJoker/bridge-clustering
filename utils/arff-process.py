import arff
import pandas as pd
from io import StringIO


def remove_first_attribute(arff_file_path, output_file_path=None):
    # 读取 ARFF 文件内容
    with open(arff_file_path, 'r') as f:
        data = arff.load(f)

    # 将原始数据转换为 DataFrame
    df = pd.DataFrame(data['data'],
                      columns=[attr[0] for attr in data['attributes']])

    # 删除第一个属性（假设它是 SequenceName）
    first_attribute_name = data['attributes'][0][0]
    df.drop(columns=[first_attribute_name], inplace=True)

    # 更新属性列表，移除第一个属性
    updated_attributes = data['attributes'][1:]

    # 更新数据，移除第一列
    updated_data = df.values.tolist()

    # 创建新的 ARFF 内容
    new_arff_content = StringIO()
    print('@RELATION', data['relation'], file=new_arff_content)

    # 打印更新后的属性
    for attr_name, attr_type in updated_attributes:
        print(f"@ATTRIBUTE {attr_name} {attr_type}", file=new_arff_content)

    print('@DATA', file=new_arff_content)

    # 打印更新后的数据行
    for row in updated_data:
        print(','.join(str(item) for item in row), file=new_arff_content)

    # 如果提供了输出文件路径，则保存到文件；否则返回字符串内容
    if output_file_path:
        with open(output_file_path, 'w') as f:
            f.write(new_arff_content.getvalue())
    else:
        return new_arff_content.getvalue()


def read_arff(file):
    with open(file, encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
    with open(file, encoding="utf-8") as f:
        df = pd.read_csv(f, header=None)
        df.columns = header
    return df


# 示例使用
folder = './data/datasets/'
real_world_folder = folder + 'real-world/'
input_file = real_world_folder + 'Yeast.arff'
output_file = real_world_folder + 'Yeast_no_seqname.arff'
# remove_first_attribute(input_file, output_file)

# 加载arff文件
with open(input_file, 'r') as f:
    decoder = arff.ArffDecoder()
    dataset = decoder.decode(f, encode_nominal=True)

# 删除第一个属性
dataset['attributes'] = dataset['attributes'][1:]
for i in range(len(dataset['data'])):
    dataset['data'][i] = dataset['data'][i][1:]

# 写入新的arff文件
with open(output_file, 'w') as f:
    encoder = arff.ArffEncoder()
    encoder.write(f, dataset)
