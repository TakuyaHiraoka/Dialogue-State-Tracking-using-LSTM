import argparse, sys, ontology_reader, dataset_walker, time, json
from fuzzywuzzy import fuzz

class BaselineTracker(object):
    def __init__(self, tagsets):
        self.tagsets = tagsets
        self.frame = {}
        self.memory = {}
        self.reset()
    def addUtter(self, utter):
        output = {'utter_index': utter['utter_index']}

        topic = utter['segment_info']['topic']
        transcript = utter['transcript'].replace('Singapore', '')

        if utter['segment_info']['target_bio'] == 'B':
            self.frame = {}
            
        if topic in self.tagsets:
            for slot in self.tagsets[topic]:
                for value in self.tagsets[topic][slot]:
                    ratio = fuzz.partial_ratio(value, transcript)
                    if ratio > 80:
                        if slot not in self.frame:
                            self.frame[slot] = []
                        if value not in self.frame[slot]:
                            self.frame[slot].append(value)
            if topic == 'ATTRACTION' and 'PLACE' in self.frame and 'NEIGHBOURHOOD' in self.frame and self.frame['PLACE'] == self.frame['NEIGHBOURHOOD']:
                del self.frame['PLACE']

            output['frame_label'] = self.frame
        return output

    def reset(self):
        self.frame = {}

def main(argv):
    print argv
    
    parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')
    parser.add_argument('--trackfile',dest='trackfile',action='store',required=True,metavar='JSON_FILE', help='File to write with tracker output')
    parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,help='JSON Ontology file')

    #args = parser.parse_args()
    args = parser.parse_args(argv)
    dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=False)
    tagsets = ontology_reader.OntologyReader(args.ontology).get_tagsets()

    track_file = open(args.trackfile, "wb")
    track = {"sessions":[]}
    track["dataset"]  = args.dataset
    start_time = time.time()

    tracker = BaselineTracker(tagsets)
    for call in dataset:
        this_session = {"session_id":call.log["session_id"], "utterances":[]}
        tracker.reset()
        for (utter,_) in call:
            sys.stderr.write('%d:%d\n'%(call.log['session_id'], utter['utter_index']))
            tracker_result = tracker.addUtter(utter)
            if tracker_result is not None:
                this_session["utterances"].append(tracker_result)
        track["sessions"].append(this_session)
    end_time = time.time()
    elapsed_time = end_time - start_time
    track['wall_time'] = elapsed_time

    json.dump(track, track_file, indent=4)

    track_file.close()

if __name__ =="__main__":
    main(sys.argv)
