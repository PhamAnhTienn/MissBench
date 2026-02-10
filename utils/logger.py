import logging
import sys


def get_console_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[ %(asctime)s ] line:%(lineno)d %(name)s - %(levelname)s - %(message)s"
        )
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger

def get_comet_logger(exp_id, args, project_name=None):
    if not args.comet:
        return None
    from comet_ml import Experiment
    comet_logger = Experiment(
        api_key=args.comet_api,
        project_name=args.project_name if project_name is None else project_name,
        workspace=args.comet_workspace,
        auto_param_logging=False,
    )
    logged_args = {
        'model': args.base_model,
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'modal_MR': args.modal_MR,
        'seed': args.seed,
        'modal': args.modal,
    }
    comet_logger.log_parameters(logged_args)
    mr = '_'.join([str(x) for x in args.modal_MR])
    comet_logger.set_name(f"{args.base_model}_{args.seed}_{mr}")
    return comet_logger



def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


# Custom Exception
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message



if __name__ == "__main__":
    logger = get_console_logger()
    logger.info("Starting")
    try:
        a = 1 / 0
    except Exception as e:
        raise CustomException(e, sys)