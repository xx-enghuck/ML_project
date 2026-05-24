from datasets import load_dataset

def main():

    # 1. 데이터셋 다운로드

    dataset = load_dataset("flwrlabs/office-home")

    # 2. 기본 구조 확인

    print(dataset)

    data = dataset["train"]

    # 3. 첫 번째 샘플 확인

    print("\nFirst sample:")

    print(data[0])

    # 4. 컬럼 확인

    print("\nColumns:")

    print(data.column_names)

    # 5. 도메인 확인

    domains = sorted(set(data["domain"]))

    print("\nDomains:")

    print(domains)

    # 6. 도메인별 개수 확인

    print("\nDomain counts:")

    for domain in domains:

        count = sum(1 for d in data["domain"] if d == domain)

        print(domain, count)

    # 7. 클래스 개수 확인

    labels = sorted(set(data["label"]))

    print("\nNumber of classes:")

    print(len(labels))

    print("\nDownload and check complete.")

if __name__ == "__main__":

    main()