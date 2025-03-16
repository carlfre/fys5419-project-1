import csv

def write_to_csv(cols: list[list], headers: list[str], filename: str):
    rows = zip(*cols) # Transpose columns to rows

    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)




if __name__ == "__main__":
    write_to_csv([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ['a', 'b', 'c'], 'output/test.csv')