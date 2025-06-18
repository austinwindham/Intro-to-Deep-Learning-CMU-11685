'''
Specification:
This utility function verifies and displays key information about a PyTorch dataloader:

1. Basic Statistics:
   - Shows which data partition is being used (train/val/test)
   - Displays number of batches and batch size
   - Reports maximum sequence lengths

2. Shape Verification:
   - Checks first batch to display tensor shapes
   - Handles both ASR datasets (with features) and Text datasets
   - Shows shapes for:
     * Features (if ASR)
     * Shifted transcripts (input sequences)
     * Golden transcripts (target sequences)
     * Length information

3. Dataset Properties:
   - Reports maximum feature length (for ASR)
   - Reports maximum transcript length
   - Shows average characters per token (for perplexity calculation)

Usage:
- Called after dataloader creation to verify proper setup
- Helps debug shape mismatches and configuration issues
- Confirms proper padding and batching behavior
'''

def verify_dataloader(dataloader):
    '''
    Computes the maximum sequence lengths of features and text in the dataloader.  
    Prints some basic statistics about the dataloader.
    '''
    def print_shapes(feats, transcripts_shifted=None, transcripts_golden=None, feat_lengths=None, transcript_lengths=None):
        if feats is not None:
            print(f"{'Feature Shape':<25}: {list(feats.shape)}")
        if transcripts_shifted is not None:
            print(f"{'Shifted Transcript Shape':<25}: {list(transcripts_shifted.shape)}")
        if transcripts_golden is not None:
            print(f"{'Golden Transcript Shape':<25}: {list(transcripts_golden.shape)}")
        if feat_lengths is not None:
            print(f"{'Feature Lengths Shape':<25}: {list(feat_lengths.shape)}")
        if transcript_lengths is not None:
            print(f"{'Transcript Lengths Shape':<25}: {list(transcript_lengths.shape)}")

    print("="*50)
    print(f"{'Dataloader Verification':^50}")
    print("="*50)
    print(f"{'Dataloader Partition':<25}: {dataloader.dataset.partition}")
    print("-"*50)
    print(f"{'Number of Batches':<25}: {len(dataloader)}")
    print(f"{'Batch Size':<25}: {dataloader.batch_size}")
    print("-"*50)
    print(f"{'Checking shapes of the data...':<50}\n")

    for i, batch in enumerate(dataloader):
        if i > 0:
            break

        if hasattr(dataloader.dataset, 'feat_max_len'):
            # ASR Dataset
            feats, transcripts_shifted, transcripts_golden, feat_lengths, transcript_lengths = batch
            print_shapes(feats, transcripts_shifted, transcripts_golden, feat_lengths, transcript_lengths)
            max_feat_len = dataloader.dataset.feat_max_len
            max_transcript_len = dataloader.dataset.text_max_len
        elif hasattr(dataloader.dataset, 'text_max_len'):
            # Text Dataset
            transcripts_shifted, transcripts_golden, lengths = batch
            print_shapes(None, transcripts_shifted, transcripts_golden, None, lengths)
            max_transcript_len = dataloader.dataset.text_max_len

    print("-"*50)
    if 'max_feat_len' in locals():
        print(f"{'Max Feature Length':<25}: {max_feat_len}")
    print(f"{'Max Transcript Length':<25}: {max_transcript_len}")
    if hasattr(dataloader.dataset, 'get_avg_chars_per_token'):
        print(f"{'Avg. Chars per Token':<25}: {dataloader.dataset.get_avg_chars_per_token():.2f}")
    print("="*50)