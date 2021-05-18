import os
import pickle
import torch


def train_model(setup):

    model = setup['model']
    optimizer = setup['optimizer']
    criterion = setup['criterion']
    scheduler = setup['scheduler']
    early_stop = setup['early_stop']
    batch_first = setup['batch_first']
    
    err_out = {}

    if torch.cuda.is_available():
        model.cuda()
        print('Using GPU')

    prev_best_loss = 1e9
    no_improvements = 0
    for epoch in range(setup['epochs']):
        total_loss = 0
        train_loss = []
        test_loss = []
        lrs = []

        model.train()

        # Loop over each subset of data
        for idx, item in enumerate(setup['train']):
            x, y = item
            x, y = x.squeeze(), y.squeeze()
            print(x.shape, y.shape)
            if not batch_first:
                print('asdf')
                x = x.transpose(0, 1)
            # Zero out the optimizer's gradient buffer
            optimizer.zero_grad()
            # Make prediction with model
            outputs = model(x)
            outputs = outputs.squeeze()

            # Compute the loss and step the optimizer
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # lrs.append(scheduler.get_last_lr()[0])
            if idx % 100 == 0:
                print('Epoch: {}, Train Loss: {}'.format(epoch, loss.item()))

            # Store loss info
            train_loss.append(loss.item())

        # Switch to evaluation mode
        model.eval()

        for idx, item in enumerate(setup['test']):
            x, y = item
            if not batch_first:
                x = x.transpose(0, 1)
            outputs = model(x)
            outputs = outputs.squeeze()

            loss = criterion(outputs, y)

            if idx % 100 == 0:
                print('Epoch: {}, Test Loss: {}'.format(epoch, loss.item()))

            # Save loss info
            total_loss += loss.item()
            test_loss.append(loss.item())

        # Save out train and test set loss.
        err_out[epoch] = {'train': train_loss, 'test': test_loss, 'lr': lrs}
        # err_out[epoch] = {'train': sum(train_loss)/len(train_loss),
        #                   'test': sum(test_loss)/len(test_loss)}

        with open(os.path.join(setup['out_dir'], 'err.p'), 'wb') as f:
            pickle.dump(err_out, f)

        # If our new loss is better than old loss, save the model. Otherwise, increment the number of times the
        # test set accuracy hasn't improved.
        if prev_best_loss > total_loss:
            torch.save(
                model.state_dict(),
                # Save new model every epoch in case the loss in the temporal generalization diverges as model trains.
                os.path.join(setup['out_dir'], 'model_{}.p'.format(epoch))
            )
            prev_best_loss = total_loss
        else:
            no_improvements += 1

        # If test set loss isn't improving, stop training the model.
        if no_improvements >= early_stop:
            break

        scheduler.step()

    del setup['scheduler']
    del setup['criterion']
    del setup['model']
    del setup['optimizer']
    del setup['train']
    del setup['test']

    return model
