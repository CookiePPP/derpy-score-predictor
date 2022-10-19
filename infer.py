from model import Model
from data import Preprocess

if __name__ == '__main__':
    checkpoint_path = "runs/2nb_2nl_256hs_0.1do_512bs_1000tags_2patience_2wf/best_val_model.pt"
    
    # get model + data processor
    model = Model.load_from_checkpoint(checkpoint_path)
    preprocess = Preprocess(model)
    
    # example input
    tag_str = "explicit, applejack, grass"
    datetime = "2015-10-16T15:51:38Z"
    scores = {
        "wilson_score": 0,
        "score": 185,
        "upvotes": 0,
        "downvotes": 0,
    }
    
    # calculate percentile
    input = preprocess(tag_str, datetime)
    output = model.infer_percentile(input, scores)[0]
    print(tag_str)
    print(f"{output['score']:.1%} Score")
    
    # for a 'safe humanized starlight glimmer' image,
    # getting 185 score would put you in the top 63% by score.
    #
    # tag_str = "safe, alternate version, artist:mauroz, part of a set, starlight glimmer, human, anime, clothes, female, humanized, sailor moon, sailor uniform, solo, staff, staff of sameness, uniform"
    # output = {'wilson_score': 0.7304874658584595, 'score': 0.6337742209434509, 'upvotes': 0.6263490915298462, 'downvotes': 0.3450292944908142}
    
    # for a `explicit, artist:shinodage, princess cadance, princess celestia, shining armor, twilight sparkle, animated, sex" image,
    # getting 185 score would put you in the bottom 9% by score.
    #
    # tag_str = "explicit, artist:deviousember, artist:shinodage, princess cadance, princess celestia, shining armor, twilight sparkle, alicorn, pony, unicorn, against wall, ahegao, animated, balls, blushing, brother and sister, clopfic in the comments, cute, eyelashes, faceless male, female, gif, incest, infidelity, leg lock, love, male, mare, nudity, offscreen character, penetration, sex, sexy armor, shiningsparkle, shipping, stallion, stallion on mare, stealth sex, straight, stupid sexy twilight, suspended congress, taint, thrill of almost being caught, tongue out, twiabetes, twilight sparkle (alicorn), underhoof, vaginal"
    # output = {'wilson_score': 0.2489914894104004, 'score': 0.09261670708656311, 'upvotes': 0.08837693929672241, 'downvotes': 0.05335080623626709}

    # for a `explicit, artist:shinodage, princess cadance, princess celestia, shining armor, twilight sparkle, animated, sex" image,
    # getting 4009 score would put you in the top 3% by score.
    # ( high scoring incest is controversial, so wilson-score is lower than expected for these tags ? )
    #
    # tag_str = "explicit, artist:deviousember, artist:shinodage, princess cadance, princess celestia, shining armor, twilight sparkle, alicorn, pony, unicorn, against wall, ahegao, animated, balls, blushing, brother and sister, clopfic in the comments, cute, eyelashes, faceless male, female, gif, incest, infidelity, leg lock, love, male, mare, nudity, offscreen character, penetration, sex, sexy armor, shiningsparkle, shipping, stallion, stallion on mare, stealth sex, straight, stupid sexy twilight, suspended congress, taint, thrill of almost being caught, tongue out, twiabetes, twilight sparkle (alicorn), underhoof, vaginal"
    # output = {'wilson_score': 0.2489914894104004, 'score': 0.9724093675613403, 'upvotes': 0.9738645553588867, 'downvotes': 0.9390099048614502}
