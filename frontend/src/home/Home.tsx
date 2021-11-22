import './Home.scss';
import {Card, Classes, EditableText, Elevation, FileInput, InputGroup, Tab, Tabs} from "@blueprintjs/core";
import {useState} from "react";
export const Home  =  () => {
    return (
        <div>
        <Card className='card'  elevation={Elevation.ONE} interactive={false}>
            Welcome to Profiler app. Add Text or File below:

            <Tabs
                id="TabsExample"
                key={"vertical" }

            >
                <Tab id="rx" title="Text" panel={<TextPanel/>}/>
                <Tab id="ng" title="File" panel={<FilePanel/>}/>
                <Tabs.Expander/>
            </Tabs>
        </Card>

        <Card className='card'  elevation={Elevation.ONE} interactive={false}>
            Output:
        </Card>
        </div>
    );
}

const TextPanel = () => {
    return (
        <EditableText
            maxLength={4000}
            maxLines={200}
            minLines={15}
            multiline={true}
            placeholder="Add Text"
        />
    )
}

const FilePanel = () => {
    const [text, setText] = useState('');
    return (
        <FileInput text={text} buttonText='Add File' />
    )
}