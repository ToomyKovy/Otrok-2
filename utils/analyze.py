from utils.plotting import col_types


def compare_with_manual(df):
    same = 0
    extended = 0
    different = 0
    labels = []
    
    for row in df.to_dict('records'):
        original_segment = row['original_segment']
        has_same = any([(row[x] or 0)>=4.0 for x in col_types.get(original_segment, [])])
    
        has_other = False
        has_any = False
        for col_type, col_list in col_types.items():
            
            for seg in col_list:
                if row[seg] >= 4.0:
                    has_any=True
                    
            if col_type == original_segment:
                continue
            for seg in col_list:
                if row[seg] >= 4.0:
                    has_other=True
                    break
    
        if original_segment not in col_types and not has_any:
            same += 1
            labels.append('same')
            continue
            
        if has_same and has_other:
            extended += 1
            labels.append('extended')
            continue
        if has_same:
            same += 1
            labels.append('same')
            continue
            
        different += 1
        labels.append('different')
    
    return {
        'same_ratio': same / len(df),
        'extended_ratio': extended / len(df),
        'different_ratio': different / len(df),
    }
