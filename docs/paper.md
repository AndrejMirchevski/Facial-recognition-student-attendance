Depth-Aware Facial Recognition for Browser-Based Student Attendance and
Exam Identity Verification

1<sup>st</sup> Konstantin Pandilovski

*School of Computer Science and Information Technology*  
*University American College Skopje  
Skopje, North Macedonia*

<pandilovskikonstantin@gmail.com>

/to add more/

> 2<sup>nd</sup> Andrej Mircevski

*School of Computer Science and Information Technology*  
*University American College Skopje  
Skopje, North Macedonia*

amircevski2005@gmail.com

*Abstract*—Manual and card-based attendance systems remain
time-consuming and vulnerable to proxy attendance, in which one student
signs in on behalf of another. Facial recognition offers an automated
alternative, but recognition systems built on ordinary webcam images are
themselves vulnerable to presentation attacks using printed photographs
or images replayed on a second screen. This paper presents a
browser-based attendance and examination identity-verification system
that combines facial recognition with a depth perception check to
distinguish a live, three-dimensional face from a flat spoofing
artifact. A student authenticates by entering their credentials in a web
interface, after which the system automatically captures five facial
frames, verifies identity against a stored biometric template, and
records attendance only once liveness and identity checks both succeed.
The system further applies duplicate-submission detection to prevent
repeated or fraudulent records, stores attendance and biometric data in
a backend database, and exposes a separate professor interface for
reviewing verified attendance. We describe the system architecture, the
verification workflow, and the security mechanisms that support its use
for both lecture attendance and pre-examination identity confirmation.

Keywords—facial recognition; attendance verification; depth perception;
anti-spoofing; presentation attack detection; biometric authentication;
Flask; web application

# Introduction

Attendance tracking in university lectures is still commonly performed
through verbal roll call, paper sign-in sheets, or card- and RFID-based
systems. These methods are time-consuming, take up part of the class
period, and remain vulnerable to proxy attendance, in which one student
signs in or scans a card on behalf of an absent classmate \[1\]. Facial
recognition has emerged as an alternative that ties attendance directly
to a student’s identity rather than to a token that can be transferred
between people, and recent systems have used both classical descriptors
and convolutional neural networks to automate this process in classroom
settings \[1\], \[2\].

Facial recognition brings its own vulnerability, however: a camera-based
system can be deceived by a presentation attack, in which a printed
photograph, a paper mask, or a video replayed on a second screen is held
up to the camera in place of a live face \[3\]. This risk is not limited
to lecture attendance. Confirming that the person seated at a device is
in fact the enrolled student is equally important before an online or
hybrid examination, where impersonation would compromise academic
integrity in the same way that proxy attendance compromises a lecture
roll \[6\]. A verification system intended for both settings must
therefore resist presentation attacks rather than perform recognition
alone.

Depth perception is one of the more reliable countermeasures to this
class of attack. A printed photograph or a screen is, geometrically, a
flat plane, while a live face has a continuous three-dimensional
surface; estimating a depth map from the captured frames, or comparing
multiple frames for the parallax a flat object cannot produce, allows a
system to reject planar spoofing artifacts that would otherwise pass a
purely appearance-based check \[4\], \[5\]. Combining this depth check
with facial verification narrows the gap between a system that merely
recognizes a face and one that confirms a live person is present.

This paper presents a browser-based attendance and identity-verification
system that applies these ideas to a university setting. Its
contributions are threefold: (1) a Flask-based web system in which a
student’s own browser captures five facial frames automatically upon
login, requiring no dedicated hardware beyond a standard webcam; (2) an
authentication pipeline that couples facial recognition with a
depth-based liveness check to reject printed-photo and screen-replay
attacks before an attendance record is written; and (3)
duplicate-detection and record-validation mechanisms, paired with
separate student and professor interfaces, that keep the resulting
attendance log auditable and resistant to repeated or fraudulent
submissions.

The rest of the paper is structured as follows. Section II provides
background and literature review on face recognition attendance systems
and depth-based anti-spoofing techniques. Section III presents the
design of the system. Section IV provides preliminary testing results of
the system. Section V covers limitations and future work. Section VI is
the conclusion.

# Related Work

Face-recognition attendance systems have been studied extensively as a
replacement for manual and card-based methods. A systematic review of
university deployments found that convolutional neural network (CNN)
based recognizers consistently outperform earlier descriptors such as
Local Binary Pattern Histograms (LBPH) in accuracy and stability under
lighting and pose variation \[1\]. Most of these systems, however, are
designed around a fixed classroom camera that performs group-level
recognition of everyone present, rather than an explicit, per-user check
initiated by the student. Design-science work on automating attendance
in human-resource and educational settings has likewise emphasized
computer-vision pipelines built around a shared camera or kiosk rather
than a browser-initiated capture flow \[2\]. The system described here
instead performs a single, per-student verification triggered from the
student’s own device, which removes the dependency on dedicated
classroom hardware and ties each capture to a specific credential entry.

A separate line of work addresses the vulnerability of camera-based face
recognition to presentation attacks. A recent survey categorizes
presentation-attack-detection (PAD) methods into behavior-based
approaches that rely on cues such as blinking or head motion,
texture-based approaches that analyze surface statistics of a printed or
displayed image, and hardware- or depth-based approaches that use the
geometry of the captured scene \[3\]. Depth-based methods are
particularly effective against printed-photo and screen-replay attacks
because both attack media are, by construction, planar, whereas a live
face is not. Prior work has shown that estimating a depth map from RGB
frames and fusing it with appearance features improves discrimination
between live and spoofed faces across several public presentation-attack
datasets \[4\], and that depth cues remain informative even when derived
from a single generic camera rather than a dedicated depth sensor \[5\].
The system in this paper follows this depth-based direction, applying a
depth check across the five captured frames as a gate before facial
verification is accepted.

Identity verification has also been studied specifically in the context
of online examinations, where automated proctoring systems monitor gaze,
active window focus, and phone use throughout a session to detect
cheating behavior \[6\]. That line of work addresses continuous
behavioral monitoring during an exam, which is a broader and more
invasive problem than the identity check considered here. The system in
this paper targets the narrower verification step, at sign-in, that both
lecture attendance and exam entry share, and is intended to complement
rather than replace full proctoring where the latter is required.

In general, previous solutions for face recognition attendance
verification, depth-based anti-spoofing, and exam session monitoring
have been considered separate issues. This paper proposes a solution
that incorporates browser-based identity verification using depth-based
liveness detection with an anti-duplicate detection mechanism. It is
intended to be used as a lightweight identity verification procedure in
both lecture and exam environments.

# Methodology and System Architecture

## System Overview

Fig. 1 shows the Architecture of the system is presented in Fig. 1. A
student accesses the web application via a typical browser and provides
his/her authentication credentials (ID and other identifiers, if
necessary). As a result, a facial verification procedure starts
immediately, during which five facial frames are captured with the use
of a camera and sent to the Flask backend for processing. A facial
embedding and depth consistency score are extracted from the provided
images, the facial embedding is compared to the student’s biometric
template and, in case the face is recognized and depth liveness test is
successful, an attendance log is written to the database along with the
timestamp.

<img src="media/image1.png" style="width:3.3in;height:2.19334in" />

1.  System architecture: browser-based student and professor clients,
    the Flask backend, the facial verification and depth anti-spoofing
    modules, and the attendance database.

## Facial Verification Pipeline

Face detection and embedding extraction follow the two-stage pattern
common to landmark-based recognition pipelines: a detector locates the
face region and facial landmarks in each frame, and an embedding model
maps the aligned face into a compact feature vector that can be compared
by distance rather than by raw pixels, following the approach introduced
for large-scale face verification and clustering \[7\]. The system
relies on this class of technique, implemented using the dlib toolkit
\[8\] for landmark detection and OpenCV \[9\] for image capture and
preprocessing, both of which are widely used in browser- and server-side
face-recognition pipelines. Capturing five frames rather than a single
image reduces the chance that one poorly lit or blurred frame causes a
false rejection; the resulting embeddings are aggregated before
comparison against the student’s enrolled template, and a match is
accepted only if the aggregated distance falls under a configured
threshold.

<img src="media/image2.png" style="width:3.3in;height:2.65445in" />

2.  Verification pipeline for a single attendance attempt, from
    five-frame capture through the identity and liveness gate to the
    attendance record.

## Depth-Based Anti-Spoofing

Alongside recognition, the system evaluates a depth-consistency signal
across the same five frames to reject printed-photo and screen-replay
attacks, following the depth-based presentation-attack-detection
approach described in \[4\] and \[5\]. Because a photograph or a screen
is planar, the small viewpoint changes and parallax present across
consecutive frames of a live face are absent or inconsistent in a
spoofed capture; the system treats this signal as a required condition
alongside facial similarity, rather than as an optional or secondary
check, so that a high-similarity match captured from a photograph is
rejected rather than merely flagged.

## Backend and Data Storage

The backend is implemented with Flask \[10\], which exposes endpoints
for credential submission, frame upload, verification, and attendance
retrieval. Verified attendance records, together with the biometric
templates required for future comparisons, are stored in a backend
database. Because facial templates are biometric data, access to the
database is restricted to the components of the system that require it,
and templates are stored separately from the plaintext attendance log
where practical, a separation that also supports the retention and
access-control obligations discussed in Section V.

## Duplicate Detection and Validation

To keep the attendance log auditable, the system checks each
verification attempt against the current session before writing a
record: a student who has already been marked present for a given
lecture or examination cannot generate a second record by repeating the
capture, and a verification attempt whose matched identity does not
correspond to the submitted credentials is rejected rather than silently
recorded under the wrong name. These checks are intended to prevent both
accidental duplicate submissions and deliberate attempts to record
attendance for another student.

## User Interfaces

The system exposes two interfaces. The student interface accepts
credential entry and then drives the automatic capture-and-verification
sequence, giving the student immediate feedback if a retry is needed.
The professor interface presents the resulting attendance records for a
given session, including verified attendees and timestamps, and allows a
professor to review and manage the record without direct access to the
underlying biometric templates.

# Results

The system was exercised in a preliminary, small-scale pilot to
characterize recognition accuracy, resistance to presentation attacks,
and processing latency before broader testing. Table I summarizes the
metrics to be reported.

1.  Preliminary Evaluation Metrics (Placeholder — Replace Before
    Submission)

| Metric                                     | Value                                    |
|--------------------------------------------|------------------------------------------|
| Verified identification accuracy           | <span class="mark">\[XX\]%</span>        |
| Printed-photo attack rejection rate        | <span class="mark">\[XX\]%</span>        |
| Screen-replay attack rejection rate        | <span class="mark">\[XX\]%</span>        |
| False rejection rate (legitimate students) | <span class="mark">\[X\]%</span>         |
| Average end-to-end verification time       | <span class="mark">\[X.X\] s</span>      |
| Pilot participants                         | <span class="mark">\[N\] students</span> |

<span class="mark">  
Once populated, this section should report how accuracy varied with
lighting condition, camera quality, and distance from the camera, and
should describe any false rejections or accepted spoofing
attempts</span>.

# Discussion

Several limitations should be considered when interpreting the design
above. Recognition and depth-consistency accuracy depend on lighting
quality and webcam resolution, both of which vary widely across student
laptops; performance under poor lighting or with low-resolution cameras
is expected to be weaker than under the pilot conditions reported in
Section IV.

The depth cues used here are estimated from ordinary RGB frames rather
than measured with a dedicated depth or infrared sensor.
Software-estimated depth of this kind is informative against planar
attacks such as printed photographs and screen replays \[4\], \[5\], but
it is a weaker signal than hardware depth sensing and may be less
effective against more sophisticated attacks, such as a well-lit
three-dimensional mask, that a stereo or time-of-flight sensor would be
better positioned to detect.

Because the system stores facial biometric templates, it is subject to
the heightened data-protection obligations that apply to biometric data
in many jurisdictions, including explicit consent, purpose limitation,
and retention-period requirements \[11\]. A deployment of this system
would need a defined retention policy for both attendance records and
biometric templates, together with a clear basis for collecting student
biometric data, before being used at institutional scale.

The evaluation in Section IV reflects a limited pilot rather than a full
validation study; broader testing across more participants, camera
types, and lighting conditions is needed before the accuracy and
spoof-rejection figures reported there can be treated as representative.

Beyond lecture attendance, the same verification step could support
identity confirmation at the start of an online or hybrid examination,
complementing rather than replacing full proctoring in settings where
continuous monitoring is required \[6\]. Future work includes evaluating
recognition accuracy across a demographically diverse student
population, adding a manual-override path for legitimate verification
failures so that no student is locked out of a lecture or exam due to a
hardware or lighting issue, and assessing whether dedicated
depth-sensing hardware, where available, meaningfully improves
resistance to more advanced spoofing attempts.

# Conclusion

This paper presented a browser-based facial recognition system for
student attendance and examination identity verification that pairs a
multi-frame verification pipeline with a depth-based anti-spoofing check
and duplicate-detection logic, exposed through separate student and
professor interfaces. By requiring both a facial match and a passed
liveness check before an attendance record is written, the system aims
to reduce both the administrative burden of manual attendance and its
vulnerability to proxy attendance and simple presentation attacks. The
preliminary results in Section IV are illustrative rather than
conclusive; validating the system across a larger, more diverse pilot,
together with addressing the privacy and hardware-dependency limitations
discussed in Section V, are the next steps toward a deployable system.

##### References

1.  A. Budiman, Fabian, R. A. Yaputera, S. Achmad, and A. Kurniawan,
    “Student attendance with face recognition (LBPH or CNN): Systematic
    literature review,” Procedia Comput. Sci., vol. 216, pp. 31–38,
    2023.

2.  B.-T. Nguyen-Tat, M.-Q. Bui, and V. M. Ngo, “Automating attendance
    management in human resources: A design science approach using
    computer vision and facial recognition,” Int. J. Inf. Manage. Data
    Insights, vol. 4, no. 1, p. 100123, 2024.

3.  Z. Ming, M. Visani, M. M. Luqman, and J.-C. Burie, “A survey on
    anti-spoofing methods for facial recognition with RGB cameras of
    generic consumer devices,” J. Imaging, vol. 6, no. 12, p. 139, Dec.
    2020.

4.  Y. Atoum, Y. Liu, A. Jourabloo, and X. Liu, “Face anti-spoofing
    using patch and depth-based CNNs,” in Proc. IEEE Int. Joint Conf.
    Biometrics (IJCB), Denver, CO, USA, 2017, pp. 319–328.

5.  Y. Wang, F. Nian, T. Li, Z. Meng, and K. Wang, “Robust face
    anti-spoofing with depth information,” J. Vis. Commun. Image
    Represent., vol. 49, pp. 332–337, 2017.

6.  Y. Atoum, L. Chen, A. X. Liu, S. D. H. Hsu, and X. Liu, “Automated
    online exam proctoring,” IEEE Trans. Multimedia, vol. 19, no. 7, pp.
    1609–1624, Jul. 2017.

7.  F. Schroff, D. Kalenichenko, and J. Philbin, “FaceNet: A unified
    embedding for face recognition and clustering,” in Proc. IEEE Conf.
    Computer Vision and Pattern Recognition (CVPR), 2015, pp. 815–823.

8.  D. E. King, “Dlib-ml: A machine learning toolkit,” J. Mach. Learn.
    Res., vol. 10, pp. 1755–1758, Dec. 2009.

9.  G. Bradski, “The OpenCV library,” Dr. Dobb’s J. Softw. Tools, vol.
    25, no. 11, pp. 120–125, Nov. 2000.

10. Pallets, “Flask.” \[Online\]. Available:
    https://flask.palletsprojects.com/

11. M. Abomhara and S. Y. Yayilgan, “An example of privacy and data
    protection best practices for biometrics data processing in border
    control: Lesson learned from SMILE,” in Intelligent Technologies and
    Applications (INTAP 2020), Communications in Computer and
    Information Science, vol. 1382, Springer, 2021, pp. 344–358.

12. 

13. 
