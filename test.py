from data import Preprocess

def inaccuracy(test_list, target_list):
    """
    l is the list being tested, l_gt is the 100% perfectly sorted list
    Function will return the average difference in index between each element in l and l_gt
    """
    assert len(test_list) == len(target_list)
    total_diff = 0
    for i in range(len(test_list)):
        total_diff += abs(test_list.index(target_list[i]) - i)
    avg_index_diff = total_diff / len(test_list)
    avg_percent_diff = avg_index_diff / len(test_list)
    return avg_percent_diff

# Sort ordered_ids.txt elements using model results and log innacuracy
def test_model(model, logger, metadata: list[dict], ordered_ids: list[str]):
    id_to_data = {str(d['id']): d for d in metadata}
    
    preprocess = Preprocess(model)
    percentiles = {} # dict of id -> {score: float, wilson_score: float, upvotes: float, downvotes: float}
    for id in ordered_ids:
        if id not in id_to_data:
            # remove from ordered_ids
            ordered_ids.remove(id)
            continue
        img_d = id_to_data[id]
        tag_str = ", ".join(img_d['rating']+img_d['characters']+img_d['ocs']+img_d['tags']) # does not include artist, art pack, series, etc.
        tag_str = tag_str.lower()
        datetime = img_d['created_at']
        scores = {
            'score'       : img_d['score'],
            'wilson_score': img_d['wilson_score'],
            'upvotes'     : img_d['upvotes'],
            'downvotes'   : img_d['downvotes'],
        }
        
        input = preprocess(tag_str, datetime)
        percentiles[id] = model.infer_percentile(input, scores)
    
    # test sorted_accuracy() with list sorted by image dsp_score
    results_sorted_by_dsp_score = sorted(ordered_ids, key=lambda x: percentiles[x]['score'])
    score_inaccuracy = inaccuracy(results_sorted_by_dsp_score, ordered_ids)
    
    # test sorted_accuracy() with list sorted by image wilson_score
    results_sorted_by_wilson_score = sorted(ordered_ids, key=lambda x: percentiles[x]['wilson_score'])
    wilson_score_inaccuracy = inaccuracy(results_sorted_by_wilson_score, ordered_ids)
    
    # test sorted_accuracy() with list sorted by image upvotes
    results_sorted_by_upvotes = sorted(ordered_ids, key=lambda x: percentiles[x]['upvotes'])
    upvotes_inaccuracy = inaccuracy(results_sorted_by_upvotes, ordered_ids)
    
    # test sorted_accuracy() with list sorted by image downvotes
    results_sorted_by_downvotes = sorted(ordered_ids, key=lambda x: percentiles[x]['downvotes'])
    downvotes_inaccuracy = inaccuracy(results_sorted_by_downvotes, ordered_ids)

    # log results and print for terminal
    loss_dict = {
        'score_inaccuracy'       : score_inaccuracy,
        'wilson_score_inaccuracy': wilson_score_inaccuracy,
        'upvotes_inaccuracy'     : upvotes_inaccuracy,
        'downvotes_inaccuracy'   : downvotes_inaccuracy,
        'best_inaccuracy'        : min(score_inaccuracy, wilson_score_inaccuracy, upvotes_inaccuracy, downvotes_inaccuracy)
    }
    logger.log(model, None, loss_dict, 'humpref_test')
    print('best_inaccuracy:', loss_dict['best_inaccuracy'])
    
    return loss_dict