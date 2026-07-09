import grapher, logger, datetime

if __name__ == "__main__":
    print(f"1 for logging, 2 for graphing")
    inp = input()
    if inp == "1":
        print(f"custom name? y/n")
        inp = input()
        if inp == "y":
            print(f"enter name")
            inp = input()
            l = logger.Logger(f"./logs/{inp}.csv")
        else:
            l = logger.Logger(f"./logs/{str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}.csv")
        l.start_logger()
    else:
        print(f"enter file name")
        inp = input()
        g = grapher.Plotter(f"./logs/{inp}.csv")
        print(f"1 for pos, 2 for vec")
        inp = input()
        if inp == "1":
            g.load_pos()
        else:
            g.load_mag()
    two = input()
