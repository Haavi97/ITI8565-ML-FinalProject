import pandas

def load_data(csv_file='water_potability.csv'):
    try: 
        return pandas.read_csv(csv_file)
    except FileNotFoundError:
        print('**************')
        print('MISSING DATA FILE')
        print('The data must be downloaded from the Kagle link in the README file')
        print('**************')
        return 0

if __name__ == '__main__':
    df = pandas.read_csv('water_potability.csv')
    print(df)