import importlib


def find_dataset_using_name(dataset_mode):
    dataset_filename = "data." + dataset_mode + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    return datasetlib


def create_dataset(args):
    """Create dataloaders
    """
    datasetlib = find_dataset_using_name(args.dataset_mode)
    dataset, embedding_dim = datasetlib.load_data(
        args.data_dir, args.dataset, args.modal_MR)

    train_loader = datasetlib.Dataloader(
        dataset['train'], batch_size=args.batch_size, embedding_dim=embedding_dim, regression=args.regression, batch_first=args.batch_first, device=args.device)
    valid_loader = datasetlib.Dataloader(
        dataset['valid'], batch_size=args.batch_size, embedding_dim=embedding_dim, regression=args.regression, batch_first=args.batch_first, device=args.device)
    test_loader = datasetlib.Dataloader(
        dataset['test'], batch_size=args.batch_size, embedding_dim=embedding_dim, regression=args.regression, batch_first=args.batch_first, device=args.device)

    print(
        f'Dataset [{str(args.dataset).upper()}, loaded as {args.dataset_mode}] is created.(batch_first: {train_loader.batch_first})')
    print(
        f'Feature dimension (A/V/L): {embedding_dim["A"]}, {embedding_dim["V"]}, {embedding_dim["L"]}')
    return [train_loader, valid_loader, test_loader], embedding_dim