#### Setup

```bash
make setup
```

> **Note**: You must put the model into `output/custom/` directory - and reference it where appropriate.

#### Run

Clone the project on a machine with a GPU (e.g. the Pluto _tracktor_ instance).
_Ensure that the instance is running before you do this step_.

Use SSH port forwarding to connect to a jupyter notebook run locally:
```bash
gcloud beta compute \
  ssh "tracktor" \
    --zone "europe-west4-a" \
    --project "pluto-a31d9" \
    -- -L 8888:localhost:8888 -Aq
```

> **Note**: The `-A` flag will forward any `ssh-add` keys to this machine. This makes it possible to authenticate over SSH with GitHub without having to generate machine specific keys. See [this guide](https://dev.to/levivm/how-to-use-ssh-and-ssh-agent-forwarding-more-secure-ssh-2c32) on how to setup SSH agent forwarding.


#### Copy video to local

Use `gcloud scp` to transfer generated avi video files to local:
```bash
gcloud beta compute \
  scp \
    --zone "europe-west4-a" \
    --project "pluto-a31d9" \
    "tracktor:/home/kalk/tracktor/experiments/scripts/*avi"
    ~/tmp/
```

> **Note**: You need to run the above command from your local machine.
