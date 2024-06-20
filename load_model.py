from Model import DBL,convlstm
def get_model(config, mode="semantic"):
    if mode == "semantic":

        if config.model == "JM":
            model = DBL.DBL(
                input_dim=10,
                num_classes=config.num_classes,
                inconv=[32, 64],
                sequence=12,
                hidden_size=88,
                input_shape=(128, 128),
                mid_conv=True,
                pad_value=config.pad_value,
            )



        elif config.model == "CF":
            model = DBL.DBL(
                input_dim=10,
                num_classes=config.num_classes,
                inconv=[32, 64],
                sequence=18,
                hidden_size=88,
                input_shape=(128, 128),
                mid_conv=True,
                pad_value=config.pad_value,
            )
        return model
    else:
        raise NotImplementedError
