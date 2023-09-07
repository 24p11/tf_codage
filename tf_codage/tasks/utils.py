class Trainer():
    
    def __init__(self, model, config, train_ds, val_ds): 
        self.model = model 
        self.config = config
        self.train_ds = train_ds 
        self.val_ds = val_ds
        
    def train(self):         
        self.model.compile(self.config.OPTIMIZER,  
                        loss=self.config.LOSS,  
                        metrics=self.config.METRICS)

        self.model.fit(self.train_ds, 
                        epochs=self.config.EPOCHS, 
                        validation_data=self.val_ds,
                        steps_per_epoch=self.config.STEPS_PER_EPOCH, 
                        validation_steps=self.config.VALIDATION_STEPS, 
                        callbacks=[self.config.CALLBACK])

    def load_model(self): 
        self.model.load_weights(self.config.PATH_MODEL).expect_partial()