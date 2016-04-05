#!/usr/bin/python3
# from PyQt5.QtWidgets import QApplication
# from models import db, Student, Directory, Exercise
from inspect import getmembers
import models
from collections import defaultdict

try:
    locale.setlocale(locale.LC_TIME, 'de_DE.utf8')
except:
    pass

models.db.connect()
tables = defaultdict(list)
for name, obj in getmembers(models, models.model_filter):
    tables[obj._meta.database].append(obj)
# tables = [obj for name, obj in getmembers(models, models.model_filter)]

for db, the_tables in tables.items():
    db.create_tables(the_tables, safe=True)
    list(map(getmembers, the_tables))
# [tables]

from cli import *
from actions import *
import locale

if not Student.select().count():
    print('No Students found.')
    # studs = import_data(path, IMPORT_PATTERN, COL_ORDER)
    # Student.insert_many(studs).execute()
    # parse_insert(path, 'student')

    # sys.exit(0)

PickStudPar = DataPar(Student.user, remember=True, lookup=False),
PickExPar = DataPar(Exercise, remember=True, lookup=False),


def pickNextData(DataClass: type):
    assert issubclass(DataClass, BaseModel)
    data_name = DataClass.__name__

    first = defaultdict(
        lambda: (lambda d, e: False),
        Exercise=lambda ex, e: not ex.completed,
        Student=lambda s, e: not GradingStatus(
            e.get('Exercise', Exercise.get()), s).completed,
    )

    def find_first(ls):
        return ls[0]

    def nextX(e):
        ls = list(DataClass)
        if not ls:
            return "There are no {}s.".format(data_name)
        try:
            d = e[data_name]
        except KeyError:
            my_first = next((n for n in ls if first[data_name](n, e)), ls[0])
            return {data_name: my_first}

        if ls[-1] == d:
            return "Last {} picked - Please Reset".format(data_name)
        news = ls[ls.index(d) + 1]
        return {data_name: news}

    def print_data(d):
        if isinstance(d, str):
            return d
        return 'picked {} {}'.format(data_name, d[data_name])

    return ENVIRONMENT_CMD() >> nextX >> print_data >> MESSAGE_CMD


cmd = CHOSE_CMD.exit(
    correct=CorrectExercise,
    correct_student=CorrectStudentExercise,
    check_all_comments=CheckAllComments,
    complete_correction=CompleteExercise,
    complete_grading=CompleteGrading,
    open_project=OpenProject,
    open_report=OpenReport,
    open_ex_files=OpenExFiles,
    validate=Validate,
    all_comments=MESSAGE_CMD << AllExComments,
    forgive_consequential_errors=ForgiveConsequentialError,
    forgive_consequential_exercise=ForgiveConsequentialExercise,
    reopen_task=ReopenTask,
    edit_grading_status=EditGradingStatus,
    add_existing_comment=AddExistingComment,
    move_comments=MoveComments,
    delete_comments=DeleteComments,
    merge_comments=MergeComments,
    comment_tasks=CommentTasks,
    comment_task=CommentTask,
    comment_exercise=AddExComment,
    reset=ResetExercisePathGrading,
    reset_noPaths=ResetExerciseGrading,
    scrape=ScrapeExercise,
    parse_ex=ParseAction,
    confirm_ex_grading=ConfirmExGrading,
    confirm_stud_grading=ConfirmGrading,
    rename_grading=RenameGrading,
    rename_app=RenameApp,
    output_grading=MESSAGE_CMD << OutputGrading,
    open_grading=OpenGrading,
    write_grading=WriteCommitGrading,
    write_gradings=WriteGradings,
    commit_gradings=CommitPreGradings,
    update_gradings_report=UpdateGradingsReport,
    update=UpdateRepo,
    update_date=UpdateRepToDate,
    clean_comments=CleanUnusedComments,
    pick_ex=DataPar(Exercise, remember=True, lookup=False),
    pick_stud=DataPar(Student.user, remember=True, lookup=False),
    next_student=pickNextData(Student),
    next_ex=pickNextData(Exercise),
    clear_stud_ex=ENVIRONMENT_DEL_CMD(Exercise, Student),
    print_environment=PRINT_ENV_CMD,
    delete_from_environment=ENV_KEY_PARAMETER >> ENVIRONMENT_DEL_CMD,
    add_grading_comment=AddGradingComment,
    add_new_comment=AddNewComment,
    backup_db=BackupDB,
    backup_restore_db=BackupRestoreDB,
)

# grammars = [(path.splitext(path.basename(p))[0], p) for p in
#             glob('grammar/*.ebnf')]
# parse_cmd_dict = {k: F_CMD(parse_insert, TypePar('str', 'path'), k) for k, p in
#                   grammars}


if __name__ == '__main__':
    # CLI_CMD.inject(({'Student': Student.get()}, Validate)).execute()
    # CLI_CMD.inject(Validate).execute()
    pickNextData(Exercise).execute()
    pickNextData(Student).execute()
    cmd.execute()
    # CHOSE_CMD.exit(**parse_cmd_dict).execute()

# app = QApplication(sys.argv)
# gui = Gui()
# app.setActivationWindow(gui)
# sys.exit(app.exec_())
